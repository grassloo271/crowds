"""
Training script for pedestrian force models.

Trains a force model (``TrueForceNet`` or ``ForceNet``) to predict multi-step
trajectories from current positions and velocities using the social force
framework.  The model is rolled out for ``rollout_horizon`` Euler steps:

    v_{t+1} = v_t + (F_goal + sum F_rep) * dt
    x_{t+1} = x_t + v_{t+1} * dt

and the objective minimizes the mean squared position and velocity error
over the full rollout, averaged over a batch of (time, person) samples.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
import argparse

from config import Config
from functions import ForceNet
from functions import TrueForceNet
from dataloader import data_loader_tp_rollout
from serialization import save


def single_rollout_loss_fn(model, pedestrian_idx, init_pos, init_vel,
                           others_positions, others_velocities,
                           target_positions, target_velocities, dt):
    """Compute mean squared trajectory error over an H-step rollout.

    At each step, the model's own predicted position/velocity are fed forward
    (true rollout), while other pedestrians' states come from ground truth.

    Args:
        model: Force model (``ForceNet`` or ``TrueForceNet``).
        pedestrian_idx: Scalar index of the focal pedestrian.
        init_pos: Starting position at time *t*, shape ``[2]``.
        init_vel: Starting velocity at time *t*, shape ``[2]``.
        others_positions: Ground-truth positions of all other pedestrians
            at times *t .. t+H-1*, shape ``[H, N-1, 2]``.
        others_velocities: Ground-truth velocities of all other pedestrians
            at times *t .. t+H-1*, shape ``[H, N-1, 2]``.
        target_positions: Ground-truth focal positions at *t+1 .. t+H*,
            shape ``[H, 2]``.
        target_velocities: Ground-truth focal velocities at *t+1 .. t+H*,
            shape ``[H, 2]``.
        dt: Simulation timestep.

    Returns:
        Scalar mean squared error (position + velocity) over the rollout.
    """
    def step_fn(carry, inputs):
        pos, vel = carry
        others_pos_h, others_vel_h, tgt_pos_h, tgt_vel_h = inputs

        rel_disp = pos - others_pos_h
        rel_vel = vel - others_vel_h
        f = jax.vmap(model.pedestrian_force, in_axes=(0, 0))(rel_disp, rel_vel)
        goal_f = model.goal_force(pedestrian_idx, vel)

        new_vel = vel + (goal_f + jnp.sum(f, axis=0)) * dt
        new_pos = pos + new_vel * dt

        step_loss = jnp.sum((new_vel - tgt_vel_h) ** 2) + jnp.sum((new_pos - tgt_pos_h) ** 2)
        return (new_pos, new_vel), step_loss

    _, step_losses = jax.lax.scan(
        step_fn,
        (init_pos, init_vel),
        (others_positions, others_velocities, target_positions, target_velocities),
    )
    return jnp.mean(step_losses)


def batch_loss_fn(model, pedestrian_indices, init_positions, init_velocities,
                  others_positions, others_velocities,
                  target_positions, target_velocities, dt):
    """Mean rollout loss over a batch of (time, person) samples."""
    loss_fn = jax.vmap(single_rollout_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None))
    return jnp.mean(loss_fn(model, pedestrian_indices, init_positions, init_velocities,
                            others_positions, others_velocities,
                            target_positions, target_velocities, dt))


@eqx.filter_jit
def make_step(model, batch, dt, opt_state, opt_update):
    """Perform one gradient-descent step on a rollout batch.

    Returns:
        Tuple of ``(loss, updated_model, updated_opt_state)``.
    """
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model,
                          batch["person_index"], batch["init_pos"], batch["init_vel"],
                          batch["others_pos"], batch["others_vel"],
                          batch["target_pos"], batch["target_vel"], dt)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_jit
def eval_step(model, batch, dt):
    """Compute batch rollout loss without gradient computation (for evaluation)."""
    return batch_loss_fn(model,
                         batch["person_index"], batch["init_pos"], batch["init_vel"],
                         batch["others_pos"], batch["others_vel"],
                         batch["target_pos"], batch["target_vel"], dt)


def main(cfg: Config):
    """Run the full training pipeline.

    Steps:
        1. Initialize the model and Adam optimizer.
        2. Load the dataset and split into train/eval by time.
        3. For each epoch, iterate over shuffled (time, person) batches and
           update model parameters.
        4. Periodically evaluate on the held-out time window.
        5. Save losses and the trained model to disk.

    Args:
        cfg: Training configuration (hyperparameters, paths, etc.).
    """

    key = jr.PRNGKey(cfg.seed)
    model_key, train_key = jr.split(key)
    if cfg.init_goal_vel_path is not None:
        goal_velocities = jnp.load(cfg.init_goal_vel_path)
    else:
        goal_velocities = jnp.zeros((cfg.num_pedestrians, 2))

    if cfg.model_type == "forcenet":
        model = ForceNet(model_key, goal_velocities, cfg.pedestrian_hidden_sizes, cfg.goal_hidden_sizes)
    elif cfg.model_type == "trueforcenet":
        model = TrueForceNet(goal_velocities, tau=jnp.array(0.0), A=jnp.array(0.0), d0=jnp.array(0.0), B=jnp.array(0.0))
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Choose 'forcenet' or 'trueforcenet'.")
    opt = optax.adam(learning_rate=cfg.learning_rate, b1=cfg.beta1, b2=cfg.beta2)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    training_losses = []
    evaluation_losses = []

    dataset = jnp.load(cfg.dataset_path)

    assert cfg.dt == dataset["dt"]

    # Skip the first 100 timesteps (transient warm-up period)
    positions = dataset["positions"][100:]
    velocities = dataset["velocities"][100:]

    # Split into train and eval sets along the time axis (no shuffling of time)
    num_timesteps = positions.shape[0]
    num_eval = int(num_timesteps * cfg.eval_fraction)
    num_train = num_timesteps - num_eval

    train_idx = jnp.arange(num_train)
    eval_idx = jnp.arange(num_train, num_timesteps)

    train_positions = positions[train_idx]
    train_velocities = velocities[train_idx]
    eval_positions = positions[eval_idx]
    eval_velocities = velocities[eval_idx]

    H = cfg.rollout_horizon
    assert num_train > H, (
        f"Training split has {num_train} timesteps but rollout_horizon={H} "
        f"requires at least {H+1}. Decrease rollout_horizon or eval_fraction."
    )
    assert num_eval > H, (
        f"Eval split has {num_eval} timesteps but rollout_horizon={H} "
        f"requires at least {H+1}. Increase eval_fraction or decrease rollout_horizon."
    )

    for i in range(cfg.num_epochs):

        train_key, val_key = jr.split(train_key)
        train_loader = data_loader_tp_rollout(
            train_positions, train_velocities,
            horizon=H, batch_size=cfg.batch_size,
            rng_key=train_key, shuffle=True, drop_last=True)

        for batch in train_loader:
            _, model, opt_state = make_step(
                model, batch, cfg.dt, opt_state, opt.update)

        if i % cfg.log_interval == 0:
            train_loader_eval = data_loader_tp_rollout(
                train_positions, train_velocities,
                horizon=H, batch_size=cfg.batch_size,
                rng_key=train_key, shuffle=False, drop_last=False)

            train_losses = []
            for batch in train_loader_eval:
                train_loss = eval_step(model, batch, cfg.dt)
                train_losses.append(train_loss)

            avg_train_loss = jnp.mean(jnp.stack(train_losses))
            training_losses.append(avg_train_loss)
            print(f"Epoch {i}, Training Loss: {avg_train_loss}")

        if i % cfg.eval_interval == 0:
            eval_loader = data_loader_tp_rollout(
                eval_positions, eval_velocities,
                horizon=H, batch_size=cfg.batch_size,
                rng_key=val_key, shuffle=False, drop_last=False)

            eval_losses = []
            for batch in eval_loader:
                eval_loss = eval_step(model, batch, cfg.dt)
                eval_losses.append(eval_loss)

            avg_eval_loss = jnp.mean(jnp.stack(eval_losses))
            evaluation_losses.append(avg_eval_loss)
            print(f"Epoch {i}, Evaluation Loss: {avg_eval_loss}")

    jnp.save(f"training_losses_{cfg.experiment_name}.npy", training_losses)
    jnp.save(f"evaluation_losses_{cfg.experiment_name}.npy", evaluation_losses)
    save(f"model_{cfg.experiment_name}.eqx", cfg, model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train pedestrian dynamics model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_pedestrians", type=int, default=200)
    parser.add_argument("--eval_fraction", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--dataset_path", type=str, default="pedestrians.npz")
    parser.add_argument("--init_goal_vel_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_horizon", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="trueforcenet",
                        choices=["forcenet", "trueforcenet"])


    args = parser.parse_args()

    cfg = Config(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        num_pedestrians=args.num_pedestrians,
        eval_fraction=args.eval_fraction,
        dt=args.dt,
        experiment_name=args.experiment_name,
        dataset_path=args.dataset_path,
        init_goal_vel_path=args.init_goal_vel_path,
        seed=args.seed,
        rollout_horizon=args.rollout_horizon,
        model_type=args.model_type,
    )

    print("Using config:")
    print(cfg)

    main(cfg)
