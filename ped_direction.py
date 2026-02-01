import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

class params:
    dt: float = 0.1

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

plt.plot(path[:,:,0], path[:,:,1])
plt.show()