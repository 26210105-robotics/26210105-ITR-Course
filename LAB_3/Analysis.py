import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Trials..csv')

# Map trial indices per algorithm
df['Trial'] = df.groupby('Control Method').cumcount() + 1
method_map = {'DLS IK': 'DLSIK', 'QPDLSIK': 'QPDLSIK', 'Mink': 'Mink'}
df['Method'] = df['Control Method'].map(method_map)

fig, axes = plt.subplots(3, 2, figsize=(15, 14))

methods = ['DLSIK', 'QPDLSIK', 'Mink']
colors = {'DLSIK': '#1f77b4', 'QPDLSIK': '#ff7f0e', 'Mink': '#2ca02c'}
markers = {'DLSIK': 'o', 'QPDLSIK': 's', 'Mink': '^'}

metrics = [
    ('pos_err_pre_pick_hover_mm', 'Pre-Pick Hover Error (mm)', axes[0, 0]),
    ('pos_err_pick_mm', 'Pick Error (mm)', axes[0, 1]),
    ('pos_err_pre_place_hover_mm', 'Pre-Place Hover Error (mm)', axes[1, 0]),
    ('pos_err_place_mm', 'Place Error (mm)', axes[1, 1]),
    ('sim_time_s', 'Simulation Time (s)', axes[2, 0]),
    ('ik_total_ms', 'IK Solver Time (ms)', axes[2, 1])
]

for col, title, ax in metrics:
    for m in methods:
        sub = df[df['Method'] == m]
        ax.plot(sub['Trial'], sub[col], label=m, color=colors[m], marker=markers[m], linewidth=1.5, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial #')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.show()
