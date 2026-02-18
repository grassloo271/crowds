from People import *
import jax.numpy as jnp
import jax
from data_manipulation import people_to_npz, render_animation_gif

crowd_a_size = 7
crowd_a = []

key = jax.random.PRNGKey(1)
key, subkey, key3, key4 = jax.random.split(key, 4)

x = jax.random.uniform(key, crowd_a_size, minval = 0, maxval= 1)
y = jax.random.uniform(subkey, crowd_a_size, minval = 0, maxval= 1) 
x1 = jax.random.uniform(key3, crowd_a_size, minval = 10, maxval= 12)
x2 = jax.random.uniform(key4, crowd_a_size, minval = 10, maxval= 12)
xy = jnp.stack([x,y], axis=-1)
goal = jnp.stack([x+5,y+9], axis=-1)
xy2 = jnp.stack([x1,x2], axis=-1)
xyz = jnp.stack([xy, xy2])

for ind in range(crowd_a_size):
    crowd_a.append(people(xy[ind], xy2[ind], xy2[ind], "tab:blue", True))

people_to_npz(crowd_a, "oop_init")

data_file = "oop_init.npz"
# render_animation_gif(data_file, 1)
