import numpy as np

filename = "data/SCENARIO_2_SESSION_2_TRIAL_5.npz"
data = np.load(filename)
positions = data["positions"]
T, N, _ = positions.shape
color = []
pos_filtered = []
orientation = data["orientations"]

def find_first_not(list, thing):
    for ind, val in enumerate(list):
        if val != thing:
            return ind

def find_last_not(list, thing):
    for ind, val in enumerate(reversed(list)):
        if val != thing:
            return len(list) - ind

for i in range(N):
    person = positions[:, i, :].copy()
    first = find_first_not(person[:, 0], 999)
    last = find_last_not(person[:, 0], 999)

    if first is not None:
        person[:first] = person[first]
    if last is not None:
        person[last:] = person[last - 1]

    for t in range(1, T):
        if person[t, 0] == 999:
            person[t] = person[t - 1]

    pos_filtered.append(person)

    # Assign color based on sum of orientations for this person
    orientation_sum = np.sum(orientation[:, i] [orientation[:, i]<900])
    color.append("red" if orientation_sum < 0 else "blue")

positions_filtered = np.stack(pos_filtered, axis=1)

output_filename = "scen_2.npz"

save_dict = {key: data[key] for key in data.files}
save_dict["x"] = positions_filtered
save_dict["color"] = np.array(color)

np.savez(output_filename, **save_dict)

print(f"Saved filtered data to {output_filename}")
print(f"Colors assigned: {color}")