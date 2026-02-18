import jax.numpy as jnp

class people:
    def __init__(self, position, velocity,goal, color,  is_goal_point=False):
        self.x = position
        self.v = velocity
        self.color = color
        self.goal = goal 
        self.is_goal = is_goal_point #determines if the goal is a point. True if is a point, false if is a velocity.

   