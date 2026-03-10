import numpy as np

file = "data/SCENARIO_1_SESSION_1_TRIAL_1.npz"
data = np.load(file)

colors = np.array(["tab:blue"]*data["positions"].shape[1])

# np.savez(
#     "s11.npz",
#     x=data["positions"],
#     color=colors
# )

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
    is_goal_pt=[]

    entry_t = []
    leave_t = []
    color = []
    
    for person in range(N):
        n = 0
        while position[n, person, 0] > 900:
            n += 1
        entry_t.append(n)
        people_init.append(position[n, person, :])
        velocity_init.append(velocity[n, person, :])
        while (position[n, person, 0] < 400 or n < 50) and n < T-1:
            n += 1
        goal_pt.append(position[n-1, person, :])
        leave_t.append(n+1)

        #point oriented
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
print(g)
np.savez(
    "s11_init.npz",
    x=a,
    v=b,
    goal=c,
    is_goal=d, 
    enter_t=e,
    color=f,
    leave_t = g
)