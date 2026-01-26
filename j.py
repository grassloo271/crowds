import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

def step(carry, t):
    x, v = carry

    new_x = 0.5*(x + v) 
    new_v = 0.5*(v - x)
    return (new_x, new_v), (new_x, new_v)

T = 100
timesteps = jnp.arange(T)
x0, v0 = (1, 0)

(xT, vT), (xs, vs) = jax.lax.scan(
        step,
        (x0,v0),
        timesteps
        )

print(xs, vs, xT, vT)
plt.plot(xs, vs)
plt.show()
