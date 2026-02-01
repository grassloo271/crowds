import jax
import jax.numpy as jnp

def semi_implicit_euler(x: jnp.ndarray, speed:jnp.ndarray, theta:jnp.ndarray, a:jnp.ndarray, alpha: jnp.ndarray, dt):
    speed_next = speed + a * dt 
    theta_next = theta + alpha * dt
    directions = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    x_next = x + dt * speed[:,None] * directions
    return x_next, speed_next, theta_next

def step_destination(carry, rng):
    x_next, speed_next, theta_next = carry
    return (x_next, speed_next, theta_next), (x_next, speed_next, theta_next)

def generate_data():
    pass 