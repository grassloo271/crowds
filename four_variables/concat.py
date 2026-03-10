import numpy as np

# Load both files
a = np.load('s11.npz')
b = np.load('s11_init_run.npz')

# Concatenate matching keys along axis 0 (stacking rows)
combined = {key: np.concat([a[key], b[key]]) for key in a.files}

# Save the result
np.savez('combined.npz', **combined)