"""
COMPLETE DATASET OVERVIEW 
(MRPC 2.5K REMOVED)
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os

# ================================================================
# OUTPUT DIRECTORY
# ================================================================
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    'rte':    '#A23B2C',   # Red
    'sst2':   '#D4A853',   # Gold
    'direct': '#555555',
}

depths = [0, 1, 2, 4, 6, 8, 10]
depth_labels = ['D', '1', '2', '4', '6', '8', '10']

RESULTS_PATH = "results"
M2_RESULTS_FILE = os.path.join(RESULTS_PATH, "m2_final_results.json")


# ================================================================
# DATA LOADING
# ================================================================
def load_results():
    """Load HKD results from JSON file."""
    if not os.path.exists(M2_RESULTS_FILE):
        print(f"⚠ Results file not found: {M2_RESULTS_FILE}")
        print("  Using hardcoded fallback values...")
        return None
    
    with open(M2_RESULTS_FILE, 'r') as f:
        return json.load(f)


def build_datasets_from_json(results):
    """Convert JSON results to dataset list for plotting."""
    datasets = []
    
    task_map = {
        'cola':       ('CoLA 8.5K', 'cola85', COLORS['cola85']),
        'cola_3.7K':  ('CoLA 3.7K', 'cola37', COLORS['cola37']),
        'cola_2.5K':  ('CoLA 2.5K', 'cola25', COLORS['cola25']),
        'mrpc':       ('MRPC 3.7K', 'mrpc37', COLORS['mrpc37']),
        'rte':        ('RTE 2.5K',  'rte',    COLORS['rte']),
        'sst2':       ('SST-2 67K', 'sst2',   COLORS['sst2']),
    }
    
    for json_key, (name, short, color) in task_map.items():
        if json_key not in results:
            print(f"  ⚠ Missing in JSON: {json_key}")
            continue
        
        r = results[json_key]
        
        # Direct KD F1
        direct_f1 = r.get('direct_kd', {}).get('f1_ci', {}).get('mean', None)
        if direct_f1 is None:
            direct_f1 = r.get('direct_kd', {}).get('test_f1_mean', None)
        
        # HKD values (depths 1, 2, 4, 6, 8, 10)
        hkd = [direct_f1]
        std = [r.get('direct_kd', {}).get('f1_std', 0.01)]
        
        ablation = r.get('depth_ablation', {})
        for d in ['1', '2', '4', '6', '8', '10']:
            if d in ablation:
                hkd.append(ablation[d].get('f1_ci', {}).get('mean', None))
                std.append(ablation[d].get('f1_std', 0.01))
            else:
                hkd.append(None)
                std.append(0.01)
        
        # Quadratic fit
        quad = r.get('quadratic_fit', {})
        
        datasets.append({
            'name': name, 'short': short, 'color': color,
            'direct': direct_f1, 'hkd': hkd, 'std': std,
            'quad': {
                'a': quad.get('a', 0.0), 'b': quad.get('b', 0.0),
                'c': quad.get('c', 0.0), 'r2': quad.get('r_squared', 0.0),
                'opt': quad.get('optimal_depth', 0.0),
            },
        })
        print(f"  ✓ {name}: Direct={direct_f1:.4f}, {len([x for x in hkd[1:] if x is not None])} HKD depths")
    
    return datasets


# ================================================================
# BUILD DATASETS
# ================================================================
print("Loading results from JSON...")
results = load_results()

if results is not None:
    datasets = build_datasets_from_json(results)
    if not datasets:
        print("⚠ No datasets loaded from JSON, using hardcoded fallback...")
        results = None

if results is None:
    # Hardcoded fallback (MRPC 2.5K REMOVED)
    datasets = [
        {
            'name': 'CoLA 8.5K', 'short': 'cola85', 'color': COLORS['cola85'],
            'direct': 0.517,
            'hkd': [0.517, 0.548, 0.546, 0.556, 0.529, 0.519, 0.419],
            'std': [0.032, 0.006, 0.014, 0.006, 0.052, 0.049, 0.019],
            'quad': {'a': -0.003041, 'b': 0.021033, 'c': 0.522926, 'r2': 0.94, 'opt': 3.46},
        },
        {
            'name': 'CoLA 3.7K', 'short': 'cola37', 'color': COLORS['cola37'],
            'direct': 0.549,
            'hkd': [0.549, 0.549, 0.557, 0.557, 0.554, 0.499, 0.484],
            'std': [0.010, 0.008, 0.007, 0.006, 0.008, 0.012, 0.015],
            'quad': {'a': -0.001657, 'b': 0.009946, 'c': 0.543045, 'r2': 0.91, 'opt': 3.00},
        },
        {
            'name': 'CoLA 2.5K', 'short': 'cola25', 'color': COLORS['cola25'],
            'direct': 0.508,
            'hkd': [0.508, 0.499, 0.503, 0.496, 0.509, 0.509, 0.478],
            'std': [0.018, 0.015, 0.012, 0.014, 0.011, 0.013, 0.016],
            'quad': {'a': -0.000860, 'b': 0.008149, 'c': 0.488625, 'r2': 0.54, 'opt': 4.74},
        },
        {
            'name': 'MRPC 3.7K', 'short': 'mrpc37', 'color': COLORS['mrpc37'],
            'direct': 0.591,
            'hkd': [0.591, 0.621, 0.602, 0.606, 0.610, 0.610, 0.564],
            'std': [0.010, 0.009, 0.007, 0.014, 0.013, 0.019, 0.057],
            'quad': {'a': -0.000997, 'b': 0.006835, 'c': 0.603467, 'r2': 0.68, 'opt': 3.43},
        },
        {
            'name': 'RTE 2.5K', 'short': 'rte', 'color': COLORS['rte'],
            'direct': 0.522,
            'hkd': [0.522, 0.521, 0.521, 0.515, 0.499, 0.521, 0.509],
            'std': [0.015, 0.017, 0.008, 0.019, 0.012, 0.018, 0.009],
            'quad': {'a': 0.000346, 'b': -0.004838, 'c': 0.526432, 'r2': 0.31, 'opt': 6.99},
        },
        {
            'name': 'SST-2 67K', 'short': 'sst2', 'color': COLORS['sst2'],
            'direct': 0.793,
            'hkd': [0.793, 0.801, 0.795, 0.798, 0.793, 0.799, 0.794],
            'std': [0.008, 0.002, 0.010, 0.005, 0.006, 0.006, 0.005],
            'quad': {'a': 0.000070, 'b': -0.001125, 'c': 0.799823, 'r2': 0.20, 'opt': 8.05},
        },
    ]
    print("  Using hardcoded fallback values")


# ================================================================
# VERIFY DATASETS
# ================================================================
print("\n" + "="*60)
print("  DATASETS LOADED")
print("="*60)
for d in datasets:
    print(f"  ✓ {d['name']}: Direct={d['direct']:.4f}, R²={d['quad']['r2']:.2f}")
print("="*60)


# ================================================================
# MAIN FIGURE: All datasets in one row
# ================================================================
print("\nGenerating complete dataset overview figure...")

fig, axes = plt.subplots(1, len(datasets), figsize=(2 * len(datasets), 2.8))
if len(datasets) == 1:
    axes = [axes]

fig.suptitle('Figure 1: Depth-performance relationship across all dataset configurations', 
             fontsize=11, fontweight='normal', y=1.02)

for i, (ax, data) in enumerate(zip(axes, datasets)):
    # Direct KD baseline
    ax.axhline(y=data['direct'], color=COLORS['direct'], linestyle='--', 
               linewidth=0.8, alpha=0.5)
    
    # HKD performance
    valid_depths = [(d, f1) for d, f1 in zip(depths[1:], data['hkd'][1:]) if f1 is not None]
    if valid_depths:
        d_vals, f1_vals = zip(*valid_depths)
        ax.plot(d_vals, f1_vals, 'o-', color=data['color'], linewidth=1.3, 
                markersize=4, markerfacecolor='white', markeredgewidth=1.0,
                markeredgecolor=data['color'], zorder=3)
    
    # Quadratic fit (only if R² > 0.6)
    if data['quad']['r2'] > 0.6:
        q = data['quad']
        x_smooth = np.linspace(0.5, 10.5, 200)
        y_smooth = q['a'] * x_smooth**2 + q['b'] * x_smooth + q['c']
        ax.plot(x_smooth, y_smooth, '-', color=data['color'], alpha=0.2, linewidth=0.8, zorder=1)
        if q['opt'] > 0:
            opt_f1 = q['a'] * q['opt']**2 + q['b'] * q['opt'] + q['c']
            ax.scatter([q['opt']], [opt_f1], color=data['color'], marker='v', s=30,
                      edgecolors='black', linewidth=0.5, zorder=10)
    
    # Error band
    hkd, std = data['hkd'], data['std']
    valid_errors = [(d, h, s) for d, h, s in zip(depths[1:], hkd[1:], std[1:]) if h is not None]
    if valid_errors:
        d_vals = [v[0] for v in valid_errors]
        h_vals = [v[1] for v in valid_errors]
        s_vals = [v[2] for v in valid_errors]
        ax.fill_between(d_vals, 
                       [h-s for h,s in zip(h_vals, s_vals)],
                       [h+s for h,s in zip(h_vals, s_vals)], 
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
    all_f1 = [x for x in data['hkd'] if x is not None]
    if all_f1:
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
plt.savefig(os.path.join(PLOTS_DIR, 'fig1_all_datasets_overview.pdf'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, 'fig1_all_datasets_overview.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ fig1_all_datasets_overview.pdf/png")


# ================================================================
# GROUPED FIGURE: CoLA variants + MRPC variants + RTE + SST-2
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), 
                         gridspec_kw={'width_ratios': [3, 1, 1, 1]})

# Panel 1: CoLA variants
ax = axes[0]
cola_datasets = [d for d in datasets if 'CoLA' in d['name']]
for data in cola_datasets:
    valid = [(d, f1) for d, f1 in zip(depths[1:], data['hkd'][1:]) if f1 is not None]
    if valid:
        d_vals, f1_vals = zip(*valid)
        ax.plot(d_vals, f1_vals, 'o-', color=data['color'], linewidth=1.2,
                markersize=4, markerfacecolor='white', markeredgewidth=0.8, label=data['name'])
    ax.axhline(y=data['direct'], color=data['color'], linestyle=':', linewidth=0.6, alpha=0.5)
ax.set_title('CoLA (data scale ablation)', fontsize=9, fontweight='normal')
ax.legend(fontsize=6, loc='lower right', frameon=False)
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)
ax.set_ylabel('Macro F1')

# Panel 2: MRPC 3.7K only
ax = axes[1]
mrpc_data = [d for d in datasets if d['name'] == 'MRPC 3.7K']
for data in mrpc_data:
    valid = [(d, f1) for d, f1 in zip(depths[1:], data['hkd'][1:]) if f1 is not None]
    if valid:
        d_vals, f1_vals = zip(*valid)
        ax.plot(d_vals, f1_vals, 'o-', color=data['color'], linewidth=1.2,
                markersize=4, markerfacecolor='white', markeredgewidth=0.8, label=data['name'])
    ax.axhline(y=data['direct'], color=data['color'], linestyle=':', linewidth=0.6, alpha=0.5)
ax.set_title('MRPC 3.7K', fontsize=9, fontweight='normal')
ax.legend(fontsize=6, loc='lower right', frameon=False)
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

# Panel 3: RTE
ax = axes[2]
rte_data = [d for d in datasets if d['short'] == 'rte']
if rte_data:
    rte_data = rte_data[0]
    valid = [(d, f1) for d, f1 in zip(depths[1:], rte_data['hkd'][1:]) if f1 is not None]
    if valid:
        d_vals, f1_vals = zip(*valid)
        ax.plot(d_vals, f1_vals, 'o-', color=rte_data['color'], linewidth=1.3,
                markersize=5, markerfacecolor='white', markeredgewidth=1.0)
    ax.axhline(y=rte_data['direct'], color=COLORS['direct'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('RTE 2.5K', fontsize=9, fontweight='normal')
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

# Panel 4: SST-2
ax = axes[3]
sst2_data = [d for d in datasets if d['short'] == 'sst2']
if sst2_data:
    sst2_data = sst2_data[0]
    valid = [(d, f1) for d, f1 in zip(depths[1:], sst2_data['hkd'][1:]) if f1 is not None]
    if valid:
        d_vals, f1_vals = zip(*valid)
        ax.plot(d_vals, f1_vals, 'o-', color=sst2_data['color'], linewidth=1.3,
                markersize=5, markerfacecolor='white', markeredgewidth=1.0)
    ax.axhline(y=sst2_data['direct'], color=COLORS['direct'], linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('SST-2 67K', fontsize=9, fontweight='normal')
ax.grid(True, alpha=0.12, linestyle='-', axis='y')
ax.set_xticks(depths); ax.set_xticklabels(depth_labels)

fig.suptitle('Figure 2: HKD performance by task family', fontsize=11, fontweight='normal', y=1.02)
plt.tight_layout(pad=0.5, w_pad=1.0)
plt.subplots_adjust(top=0.85)
plt.savefig(os.path.join(PLOTS_DIR, 'fig2_grouped_by_task.pdf'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, 'fig2_grouped_by_task.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ fig2_grouped_by_task.pdf/png")


print("\n" + "="*60)
print("  FIGURES GENERATED")
print("="*60)
print(f"  Data source: {'JSON' if results is not None else 'Hardcoded fallback'}")
print(f"  Output directory: {PLOTS_DIR}/")
print("  fig1_all_datasets_overview.pdf/png  - All datasets in one row")
print("  fig2_grouped_by_task.pdf/png         - Grouped by task family")
print("="*60)