from People import *
import jax.numpy as jnp
import jax
from data_manipulation import people_to_npz, render_animation_gif

def make_crowd(population:int, start:list, start_spread: list, goal:list, goal_spread:list, vel:list, vel_spread: list, goal_type=True, color="tab:blue", key = jax.random.PRNGKey(1)):
    k1, k2, k3, k4, k5, k6= jax.random.split(key, 6)    
    x0 = jax.random.uniform(k1, population, minval = start[0] - start_spread[0], maxval = start[0] + start_spread[0])
    y0 = jax.random.uniform(k2, population, minval =  start[1] - start_spread[1], maxval = start[1] + start_spread[1])
    xy0 = jnp.stack([x0, y0], axis = -1)
    xg = jax.random.uniform(k3, population, minval = goal[0] - goal_spread[0], maxval = goal[0] + goal_spread[0])
    yg = jax.random.uniform(k4, population, minval = goal[1] - goal_spread[1], maxval = goal[1] + goal_spread[1])
    gl = jnp.stack([xg, yg], axis = -1)
    vx = jax.random.uniform(k5, population, minval =  vel[0] - vel_spread[0], maxval = vel[0] + vel_spread[0])
    vy = jax.random.uniform(k6, population, minval = vel[1] - vel_spread[1], maxval = vel[1] + vel_spread[1])
    velocity = jnp.stack([vx, vy], axis = -1)
    peoples = []
    for ind in range(population):
        peoples.append(people(xy0[ind], velocity[ind], gl[ind], color, goal_type))

    return peoples

#=====================================3 crowd case===========================
case = 0

if case == 0:
    start = [-20,20]
    start_spread = [5,5]
    goal = [-10.0, -20.0]
    goal_spread = [5, 5]
    vel = [1, 1]
    vel_spread = [0.2, 0.2]

    b_start = [20,-20]
    b_start_spread = [5,5]
    b_goal = [-20.0, 20.0]
    b_goal_spread = [5, 5]
    b_vel = [1, 1]
    b_vel_spread = [0.2, 0.2]

    c_start = [-15,-5]
    c_start_spread = [5,5]
    c_goal = [15, 20]
    c_goal_spread = [5, 5]
    c_vel = [1, 1]
    c_vel_spread = [0.2, 0.2]

    crowd_a = make_crowd(60, start, start_spread, goal, goal_spread, vel, vel_spread)
    crowd_b = make_crowd(60, b_start, b_start_spread, b_goal, b_goal_spread, b_vel, b_vel_spread, color="tab:red")
    crowd_c = make_crowd(60, c_start, c_start_spread, c_goal, c_goal_spread, c_vel, c_vel_spread, color="tab:green", key= jax.random.PRNGKey(2))
    crowd = crowd_a + crowd_b + crowd_c
    name = "big_test_3"

#crowd = crowd_a + crowd_b

#=================================smaller 2 crowd case================================
elif case == 1:
    start = [-10,-10]
    start_spread = [0,0]
    goal = [10.0, 10.0]
    goal_spread = [0, 0]
    vel = [2, 2]
    vel_spread = [0, 0]


    b_start = [-9.5,-9.5]
    b_start_spread = [5,0]
    b_goal = [10.0, 10.0]
    b_goal_spread = [0, 0]
    b_vel = [0, 0]
    b_vel_spread = [0, 0]

    crowd_a = make_crowd(1, start, start_spread, goal, goal_spread, vel, vel_spread)
    crowd_b = make_crowd(1, b_start, b_start_spread, b_goal, b_goal_spread, b_vel, b_vel_spread, color="tab:red")
    crowd = crowd_a + crowd_b
    name = "front_coming"

elif case == 2:
    start = [-10,-10]
    start_spread = [5,5]
    goal = [10.0, 10.0]
    goal_spread = [5, 5]
    vel = [2, 2]
    vel_spread = [1, 1]

    b_start = [-9.5,-9.5]
    b_start_spread = [5,5]
    b_goal = [10.0, 10.0]
    b_goal_spread = [5, 5]
    b_vel = [0, 0]
    b_vel_spread = [0, 0]

    crowd_a = make_crowd(10, start, start_spread, goal, goal_spread, vel, vel_spread)
    crowd_b = make_crowd(10, b_start, b_start_spread, b_goal, b_goal_spread, b_vel, b_vel_spread, key = jax.random.PRNGKey(2), color="tab:red")
    crowd = crowd_a + crowd_b
    name = "front_verification"



people_to_npz(crowd, name)

# # data_file = "oop_init.npz"
# # render_animation_gif(data_file, 1)
