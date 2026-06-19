"""
M2 Paper Figures - COMPLETE DATASET OVERVIEW
All 7 configurations in a single row for direct comparison.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ================================================================
# PROFESSIONAL STYLING (LNCS COMPLIANT)
# ================================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6,
    'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.3, 'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
})

# Color palette - distinct for each dataset
COLORS = {
    'cola85': '#6A4E9B',   # Purple
    'cola37': '#2E8B57',   # Green
    'cola25': '#CD8E3C',   # Orange
    'mrpc37': '#3A6EA5',   # Blue
    'mrpc25': '#5BA3D9',   # Light blue
    'rte':    '#A23B2C',   # Red
    'sst2':   '#D4A853',   # Gold
    'direct': '#555555',
}

depths = [0, 1, 2, 4, 6, 8, 10]
depth_labels = ['D', '1', '2', '4', '6', '8', '10']


# ================================================================
# DATA
# ================================================================
datasets = [
    {
        'name': 'CoLA 8.5K', 'short': 'cola85', 'color': COLORS['cola85'],
        'direct': 0.517, 'teacher': 0.775,
        'hkd': [0.517, 0.548, 0.546, 0.556, 0.529, 0.519, 0.419],
        'std': [0.032, 0.006, 0.014, 0.006, 0.052, 0.049, 0.019],
        'quad': {'a': -0.003041, 'b': 0.021033, 'c': 0.522926, 'r2': 0.94, 'opt': 3.46},
    },
    {
        'name': 'CoLA 3.7K', 'short': 'cola37', 'color': COLORS['cola37'],
        'direct': 0.549, 'teacher': 0.757,
        'hkd': [0.549, 0.549, 0.557, 0.557, 0.554, 0.499, 0.484],
        'std': [0.010, 0.008, 0.007, 0.006, 0.008, 0.012, 0.015],
        'quad': {'a': -0.001657, 'b': 0.009946, 'c': 0.543045, 'r2': 0.91, 'opt': 3.00},
    },
    {
        'name': 'CoLA 2.5K', 'short': 'cola25', 'color': COLORS['cola25'],
        'direct': 0.508, 'teacher': 0.713,
        'hkd': [0.508, 0.499, 0.503, 0.496, 0.509, 0.509, 0.478],
        'std': [0.018, 0.015, 0.012, 0.014, 0.011, 0.013, 0.016],
        'quad': {'a': -0.000860, 'b': 0.008149, 'c': 0.488625, 'r2': 0.54, 'opt': 4.74},
    },
    {
        'name': 'MRPC 3.7K', 'short': 'mrpc37', 'color': COLORS['mrpc37'],
        'direct': 0.591, 'teacher': 0.808,
        'hkd': [0.591, 0.621, 0.602, 0.606, 0.610, 0.610, 0.564],
        'std': [0.010, 0.009, 0.007, 0.014, 0.013, 0.019, 0.057],
        'quad': {'a': -0.000997, 'b': 0.006835, 'c': 0.603467, 'r2': 0.68, 'opt': 3.43},
    },
    {
        'name': 'MRPC 2.5K', 'short': 'mrpc25', 'color': COLORS['mrpc25'],
        'direct': 0.576, 'teacher': 0.769,
        'hkd': [0.576, 0.600, 0.599, 0.590, 0.577, 0.594, 0.577],
        'std': [0.020, 0.008, 0.014, 0.018, 0.014, 0.010, 0.007],
        'quad': {'a': 0.000204, 'b': -0.004369, 'c': 0.604305, 'r2': 0.58, 'opt': 10.70},
    },
    {
        'name': 'RTE 2.5K', 'short': 'rte', 'color': COLORS['rte'],
        'direct': 0.522, 'teacher': 0.694,
        'hkd': [0.522, 0.521, 0.521, 0.515, 0.499, 0.521, 0.509],
        'std': [0.015, 0.017, 0.008, 0.019, 0.012, 0.018, 0.009],
        'quad': {'a': 0.000346, 'b': -0.004838, 'c': 0.526432, 'r2': 0.31, 'opt': 6.99},
    },
    {
        'name': 'SST-2 67K', 'short': 'sst2', 'color': COLORS['sst2'],
        'direct': 0.793, 'teacher': 0.919,
        'hkd': [0.793, 0.801, 0.795, 0.798, 0.793, 0.799, 0.794],
        'std': [0.008, 0.002, 0.010, 0.005, 0.006, 0.006, 0.005],
        'quad': {'a': 0.000070, 'b': -0.001125, 'c': 0.799823, 'r2': 0.20, 'opt': 8.05},
    },
]


# ================================================================
# MAIN FIGURE: All 7 datasets in one row
# ================================================================
print("Generating complete dataset overview figure...")

fig, axes = plt.subplots(1, 7, figsize=(14, 2.8))
fig.suptitle('Figure 1: Depth-performance relationship across all dataset configurations', 
             fontsize=11, fontweight='normal', y=1.02)

for i, (ax, data) in enumerate(zip(axes, datasets)):
    # Direct KD baseline
    ax.axhline(y=data['direct'], color=COLORS['direct'], linestyle='--', 
               linewidth=0.8, alpha=0.5)
    
    # HKD performance
    ax.plot(depths[1:], data['hkd'][1:], 'o-', color=data['color'], linewidth=1.3, 
            markersize=4, markerfacecolor='white', markeredgewidth=1.0,
            markeredgecolor=data['color'], zorder=3)
    
    # Quadratic fit (only if R² > 0.6)
    if data['quad']['r2'] > 0.6:
        q = data['quad']
        x_smooth = np.linspace(0.5, 10.5, 200)
        y_smooth = q['a'] * x_smooth**2 + q['b'] * x_smooth + q['c']
        ax.plot(x_smooth, y_smooth, '-', color=data['color'], alpha=0.2, linewidth=0.8, zorder=1)
        opt_f1 = q['a'] * q['opt']**2 + q['b'] * q['opt'] + q['c']
        ax.scatter([q['opt']], [opt_f1], color=data['color'], marker='v', s=30,
                  edgecolors='black', linewidth=0.5, zorder=10)
    
    # Error band
    hkd, std = data['hkd'], data['std']
    ax.fill_between(depths[1:], 
                    [h-s for h,s in zip(hkd[1:], std[1:])],
                    [h+s for h,s in zip(hkd[1:], std[1:])], 
                    color=data['color'], alpha=0.1)
    
    # Title with R²
    r2_text = f"R²={data['quad']['r2']:.2f}"
    if data['quad']['r2'] > 0.8:
        r2_text += " *"
    ax.set_title(f"{data['name']}\n({r2_text})", fontsize=7, fontweight='normal', 
                 loc='center', pad=3)
    
    # Grid and labels
    ax.grid(True, alpha=0.12, linestyle='-', axis='y')
    ax.set_xticks(depths)
    ax.set_xticklabels(depth_labels, fontsize=6)
    ax.set_xlabel('Depth', fontsize=7)
    
    if i == 0:
        ax.set_ylabel('Macro F1', fontsize=8)
    
    # Y-axis limits
    all_f1 = data['hkd']
    ymin = min(all_f1) - 0.03
    ymax = max(all_f1) + 0.03
    ax.set_ylim(ymin, ymax)

# Legend in last subplot
axes[-1].plot([], [], 'o-', color='gray', linewidth=1.3, markersize=4,
              markerfacecolor='white', markeredgewidth=1.0, label='HKD')
axes[-1].axhline(y=0, color=COLORS['direct'], linestyle='--', linewidth=0.8, label='Direct KD')
axes[-1].legend(fontsize=6, loc='lower right', frameon=False, title='Legend', title_fontsize=7)

plt.tight_layout(pad=0.5, w_pad=0.8)
plt.subplots_adjust(top=0.85)
plt.savefig('fig1_all_datasets_overview.pdf', dpi=600, bbox_inches='tight')
plt.savefig('fig1_all_datasets_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ fig1_all_datasets_overview.pdf/png (7 datasets in one row)")


# ================================================================
# GROUPED FIGURE: CoLA variants + MRPC variants + RTE + SST-2
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), 
                         gridspec_kw={'width_ratios': [3, 2, 1, 1]})

# Panel 1: CoLA variants (3 subplots in one)
ax = axes[0]
cola_datasets = [d for d in datasets if 'CoLA' in d['name']]
for data in cola_datasets:
    ax.plot(depths[1:], data['hkd'][1:], 'o-', color=data['color'], linewidth=1.2,
            markersize=4, markerfacecolor='white', markeredgewidth=0.8, label=data['name'])
    ax.axhline(y=data['direct'], color=data['color'], linestyle=':', linewidth=0.6, alpha=0.5)
ax.set_title('CoLA (data scale ablation)', fontsize=9, fontweight='normal')
ax.legend(fontsize=6, loc='lower right', frameon=False)
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)
ax.set_ylabel('Macro F1')

# Panel 2: MRPC variants
ax = axes[1]
mrpc_datasets = [d for d in datasets if 'MRPC' in d['name']]
for data in mrpc_datasets:
    ax.plot(depths[1:], data['hkd'][1:], 'o-', color=data['color'], linewidth=1.2,
            markersize=4, markerfacecolor='white', markeredgewidth=0.8, label=data['name'])
    ax.axhline(y=data['direct'], color=data['color'], linestyle=':', linewidth=0.6, alpha=0.5)
ax.set_title('MRPC (data scale ablation)', fontsize=9, fontweight='normal')
ax.legend(fontsize=6, loc='lower right', frameon=False)
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

# Panel 3: RTE
ax = axes[2]
rte_data = [d for d in datasets if d['short'] == 'rte'][0]
ax.plot(depths[1:], rte_data['hkd'][1:], 'o-', color=rte_data['color'], linewidth=1.3,
        markersize=5, markerfacecolor='white', markeredgewidth=1.0)
ax.axhline(y=rte_data['direct'], color=COLORS['direct'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('RTE 2.5K', fontsize=9, fontweight='normal')
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

# Panel 4: SST-2
ax = axes[3]
sst2_data = [d for d in datasets if d['short'] == 'sst2'][0]
ax.plot(depths[1:], sst2_data['hkd'][1:], 'o-', color=sst2_data['color'], linewidth=1.3,
        markersize=5, markerfacecolor='white', markeredgewidth=1.0)
ax.axhline(y=sst2_data['direct'], color=COLORS['direct'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('SST-2 67K', fontsize=9, fontweight='normal')
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

fig.suptitle('Figure 2: HKD performance by task family', fontsize=11, fontweight='normal', y=1.02)
plt.tight_layout(pad=0.5, w_pad=1.0)
plt.subplots_adjust(top=0.85)
plt.savefig('fig2_grouped_by_task.pdf', dpi=600, bbox_inches='tight')
plt.savefig('fig2_grouped_by_task.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ fig2_grouped_by_task.pdf/png (CoLA variants + MRPC variants + RTE + SST-2)")


print("\n" + "="*60)
print("  FIGURES GENERATED")
print("="*60)
print("  fig1_all_datasets_overview.pdf/png  - All 7 datasets in one row")
print("  fig2_grouped_by_task.pdf/png         - Grouped by task family")
print("="*60)