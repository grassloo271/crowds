import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

class params:
    dt: float = 0.1
    del_t: float = 1

def semi_implicit_euler(x: jnp.ndarray, speed:jnp.ndarray, theta:jnp.ndarray, a:jnp.ndarray, alpha: jnp.ndarray, p: params):
    speed_next = speed + a * p.dt 
    theta_next = theta + alpha * p.dt
    directions = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    x_next = x + p.dt * speed[:,None] * directions
    return x_next, speed_next, theta_next

def force_parallel():
    return 0.1

def force_perpendicular():
    return 1

def step_destination(carry, p: params):
    x_next, speed_next, theta_next = carry
    F_parallel = force_parallel()
    F_perp = force_perpendicular()
    x_next, speed_next, theta_next = semi_implicit_euler(x_next, speed_next, theta_next, F_parallel, F_perp, p)
    return (x_next, speed_next, theta_next), (x_next, speed_next, theta_next)

def generate_data(x0, speed0, theta0, T, p):

    def scan_body(carry, _ : float):
        return step_destination(carry, p)
    
    (xT, vT, thetaT), (xs, vs, thetas) = jax.lax.scan(scan_body, (x0, speed0, theta0), jnp.zeros(T))
    return xs

path = generate_data(jnp.array([[0.0,0.0], [1,1]]), jnp.array([1.0,2]), jnp.array([1.0,3]), 80, params)
# Helbing model - attempt

def hel_force(x: jnp.array, speed: jnp.array, theta: jnp.array, p: params):
    diff = x[:, None, :] - x[None, :, :]
    D = jnp.linalg.norm(diff, axis=-1, keepdims=True)

    vel = speed[:, None] * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) 
    a1 = diff - vel[None, :, :] * p.del_t
    a1_norm = jnp.linalg.norm(a1, axis=-1, keepdims=True)
    b = 0.5 * jnp.sqrt((jnp.reshape(D + a1_norm, [2,2])) ** 2 - (p.del_t * speed[None, :] )** 2)
    return b

def project(x: jnp.array, v_rel: jnp.array, v_goal=None):
    if jnp.linalg.norm(v_rel) == 0:
        v_rel = v_goal

    v_dir = v_rel / jnp.linalg.norm(v_rel)
    x_parallel = jnp.dot(x, v_dir)[:, None] * v_dir 
    x_perp = x - x_parallel
    return x_parallel, x_perp


