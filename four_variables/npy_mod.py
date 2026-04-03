import numpy as np
file = "data/SCENARIO_2_SESSION_2_TRIAL_5.npz"
data = np.load(file)

def find_first_not(arr, thing):
    for ind, val in enumerate(arr):
        if val != thing:
            return ind

def find_last_not(arr, thing):
    for ind, val in enumerate(reversed(arr)):
        if val != thing:
            return len(arr) - ind

def init_condition_scenario_1(data):
    position = data["positions"]
    velocity = data["velocities"]
    orientation = data["orientations"]
    T, N, _ = position.shape
    people_init = []
    velocity_init = []
    goal = []
    goal_pt = []
    is_goal = []
    is_goal_pt = []
    entry_t = []
    leave_t = []
    color = []
    
    for person in range(N):
        # Find entry: first valid position
        first = find_first_not(position[:, person, 0], 999)
        entry_t.append(first)
        people_init.append(position[first, person, :])
        velocity_init.append(velocity[first, person, :])

        # Find leave: last valid position
        last = find_last_not(position[:, person, 0], 999)
        leave_t.append(last)
        goal_pt.append(position[last - 1, person, :])

        # Point oriented
        if sum(orientation[:, person][orientation[:, person] < 900]) > 0:
            goal.append([0, 1000])
            color.append("tab:green")
        else:
            goal.append([0, -1000])
            color.append("tab:red")
        
        is_goal.append(True)
        is_goal_pt.append(True)

    return people_init, velocity_init, goal, is_goal, entry_t, color, goal_pt, is_goal_pt, leave_t

a, b, _, _, e, f, c, d, g = init_condition_scenario_1(data)
print(c)
np.savez(
    "s225_init.npz",
    x=a,
    v=b,
    goal=c,
    is_goal=d,
    enter_t=e,
    color=f,
    leave_t=g
)