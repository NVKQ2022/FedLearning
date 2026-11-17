#!/usr/bin/env python3
"""
Publication-Grade Figure V.4 Generator (Minimalist Edition):
Multi-Regime Empirical Analysis of the Stability-Plasticity Trade-Off.

Extracts 100% of the metric data dynamically from all 20 scenario experiment files
in scenarios/flower_e2_*, scenarios/flower_e3_*, scenarios/flower_e4_*, scenarios/flower_e5_*.

Generates:
  (a) Parameter drift stabilization vs. mu
  (b) Global Macro-F1 utility vs. mu
  (c) Minority class extinction cliff at mu >= 0.01
  (d) 2D Multi-regime Pareto trade-off plane
Designed for maximum clarity and minimalism (no clutter, no excessive callout boxes).
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Dynamic Data Extraction from Experiment Artifacts
# -----------------------------------------------------------------------------
regime_configs = [
    {
        'id': 'iid',
        'label': r'IID ($K=5$)',
        'color': '#1f77b4',  # Royal Blue
        'marker': 'o',
        'scenarios': [
            ('0.0 (FedAvg)', 'scenarios/flower_e2_fedavg_iid/server/metrics.json'),
            ('0.001', 'scenarios/flower_e2_fedprox_mu=0.001_iid/server/metrics.json'),
            ('0.01', 'scenarios/flower_e2_fedprox_mu=0.01_iid/server/metrics.json'),
            ('0.025', 'scenarios/flower_e2_fedprox_mu=0.025_iid/server/metrics.json'),
            ('0.05', 'scenarios/flower_e2_fedprox_mu=0.05_iid/server/metrics.json'),
        ]
    },
    {
        'id': 'alpha10',
        'label': r'Mild Non-IID ($\alpha=1.0$)',
        'color': '#2ca02c',  # Forest Green
        'marker': 'D',
        'scenarios': [
            ('0.0 (FedAvg)', 'scenarios/flower_e3_fedavg_dirichlet_alpha=1.0/server/metrics.json'),
            ('0.001', 'scenarios/flower_e3_fedprox_mu=0.001_dirichlet_alpha=1.0/server/metrics.json'),
            ('0.01', 'scenarios/flower_e3_fedprox_mu=0.01_dirichlet_alpha=1.0/server/metrics.json'),
            ('0.025', 'scenarios/flower_e3_fedprox_mu=0.025_dirichlet_alpha=1.0/server/metrics.json'),
            ('0.05', 'scenarios/flower_e3_fedprox_mu=0.05_dirichlet_alpha=1.0/server/metrics.json'),
        ]
    },
    {
        'id': 'alpha05',
        'label': r'Moderate Non-IID ($\alpha=0.5$)',
        'color': '#ff7f0e',  # Amber Orange
        'marker': 's',
        'scenarios': [
            ('0.0 (FedAvg)', 'scenarios/flower_e4_fedavg_dirichlet_alpha=0.5/server/metrics.json'),
            ('0.001', 'scenarios/flower_e4_fedprox_mu=0.001_dirichlet_alpha=0.5/server/metrics.json'),
            ('0.01', 'scenarios/flower_e4_fedprox_mu=0.01_dirichlet_alpha=0.5/server/metrics.json'),
            ('0.025', 'scenarios/flower_e4_fedprox_mu=0.025_dirichlet_alpha=0.5/server/metrics.json'),
            ('0.05', 'scenarios/flower_e4_fedprox_mu=0.05_dirichlet_alpha=0.5/server/metrics.json'),
        ]
    },
    {
        'id': 'alpha01',
        'label': r'Severe Non-IID ($\alpha=0.1$)',
        'color': '#d62728',  # Crimson Red
        'marker': '^',
        'scenarios': [
            ('0.0 (FedAvg)', 'scenarios/flower_e5_fedavg_dirichlet_alpha=0.1/server/metrics.json'),
            ('0.001', 'scenarios/flower_e5_fedprox_mu=0.001_dirichlet_alpha=0.1/server/metrics.json'),
            ('0.01', 'scenarios/flower_e5_fedprox_mu=0.01_dirichlet_alpha=0.1/server/metrics.json'),
            ('0.025', 'scenarios/flower_e5_fedprox_mu=0.025_dirichlet_alpha=0.1/server/metrics.json'),
            ('0.05', 'scenarios/flower_e5_fedprox_mu=0.05_dirichlet_alpha=0.1/server/metrics.json'),
        ]
    }
]

# Verify and load all data dynamically
experiment_data = {}
print("=== Loading Experiment Metrics Dynamically from scenarios/ ===")
for reg in regime_configs:
    reg_id = reg['id']
    experiment_data[reg_id] = {
        'mu_labels': [],
        'drift_r10': [],
        'macro_f1': [],
        'bf_recall': [],
        'web_recall': [],
        'accuracy': []
    }
    for mu_label, path in reg['scenarios']:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing experiment file: {path}")
        with open(path, 'r') as f:
            m = json.load(f)
        
        drift = float(m['round_history']['drift_l2'][-1])
        f1 = float(m['macro_f1']) * 100.0
        acc = float(m['accuracy']) * 100.0
        bf_rec = float(m['per_class_recall'].get('Brute-force', 0.0)) * 100.0
        web_rec = float(m['per_class_recall'].get('Web-based', 0.0)) * 100.0
        
        experiment_data[reg_id]['mu_labels'].append(mu_label)
        experiment_data[reg_id]['drift_r10'].append(drift)
        experiment_data[reg_id]['macro_f1'].append(f1)
        experiment_data[reg_id]['bf_recall'].append(bf_rec)
        experiment_data[reg_id]['web_recall'].append(web_rec)
        experiment_data[reg_id]['accuracy'].append(acc)

print("All 20 experiment files loaded and verified successfully!")

# -----------------------------------------------------------------------------
# 2. Minimalist Publication Plotting Configuration
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['grid.alpha'] = 0.5
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8), dpi=300)

mu_labels = ['0.0 (FedAvg)', '0.001', '0.01', '0.025', '0.05']
x_indices = np.arange(len(mu_labels))

# =============================================================================
# Subplot (a): Parameter Drift Stability vs. mu
# =============================================================================
ax_a = axes[0, 0]
for reg in regime_configs:
    d = experiment_data[reg['id']]
    ax_a.plot(x_indices, d['drift_r10'],
              marker=reg['marker'], markersize=7, linewidth=2.0,
              color=reg['color'], label=reg['label'])

ax_a.set_xticks(x_indices)
ax_a.set_xticklabels(mu_labels)
ax_a.set_xlabel(r'Proximal Regularization Coefficient $\mu$', fontweight='bold')
ax_a.set_ylabel(r'Parameter Drift ($L_2$ Norm)', fontweight='bold')
ax_a.set_title(r'(a) Parameter Drift Stability vs. $\mu$', fontweight='bold', pad=10)
ax_a.set_ylim(-0.5, 24.5)
ax_a.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
ax_a.grid(True, linestyle='--', alpha=0.5)

# =============================================================================
# Subplot (b): Global Utility (Macro-F1) vs. mu
# =============================================================================
ax_b = axes[0, 1]
for reg in regime_configs:
    d = experiment_data[reg['id']]
    ax_b.plot(x_indices, d['macro_f1'],
              marker=reg['marker'], markersize=7, linewidth=2.0,
              color=reg['color'], label=reg['label'])

ax_b.set_xticks(x_indices)
ax_b.set_xticklabels(mu_labels)
ax_b.set_xlabel(r'Proximal Regularization Coefficient $\mu$', fontweight='bold')
ax_b.set_ylabel(r'Global Macro-F1 Score (%)', fontweight='bold')
ax_b.set_title(r'(b) Global Utility (Macro-F1) vs. $\mu$', fontweight='bold', pad=10)
ax_b.set_ylim(41.0, 70.5)
ax_b.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
ax_b.grid(True, linestyle='--', alpha=0.5)

# =============================================================================
# Subplot (c): Minority Class Recall vs. mu
# =============================================================================
ax_c = axes[1, 0]

# Solid lines for Brute-force Recall
for reg in regime_configs:
    d = experiment_data[reg['id']]
    ax_c.plot(x_indices, d['bf_recall'],
              marker=reg['marker'], markersize=7, linewidth=2.0,
              color=reg['color'], label=reg['label'])

# Dashed lines for Web-based Recall
for reg in regime_configs:
    d = experiment_data[reg['id']]
    ax_c.plot(x_indices, d['web_recall'],
              marker=reg['marker'], markersize=4.5, linestyle='--', linewidth=1.2,
              color=reg['color'], alpha=0.6)

# Line style indicator proxy in legend
ax_c.plot([], [], color='black', linestyle='-', linewidth=2.0, label='Brute-force (solid)')
ax_c.plot([], [], color='black', linestyle='--', linewidth=1.2, label='Web-based (dashed)')

ax_c.set_xticks(x_indices)
ax_c.set_xticklabels(mu_labels)
ax_c.set_xlabel(r'Proximal Regularization Coefficient $\mu$', fontweight='bold')
ax_c.set_ylabel(r'Minority Class Recall (%)', fontweight='bold')
ax_c.set_title(r'(c) Minority Class Recall vs. $\mu$', fontweight='bold', pad=10)
ax_c.set_ylim(-1.0, 24.0)
ax_c.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8.8, loc='upper right')
ax_c.grid(True, linestyle='--', alpha=0.5)

# =============================================================================
# Subplot (d): 2D Multi-Regime Pareto Trade-off Plane
# =============================================================================
ax_d = axes[1, 1]

for reg in regime_configs:
    d = experiment_data[reg['id']]
    drift_vals = d['drift_r10']
    f1_vals = d['macro_f1']
    
    ax_d.plot(drift_vals, f1_vals,
              marker=reg['marker'], markersize=7, linewidth=1.8,
              color=reg['color'], label=reg['label'])
    
    # Label starting point (mu=0) and sweet spot (mu=0.001) with clean, minimal text
    ax_d.annotate(r'$\mu=0$', (drift_vals[0], f1_vals[0]),
                  textcoords='offset points', xytext=(5, 3),
                  fontsize=8.0, color=reg['color'])
    
    # Label mu=0.001 sweet spot
    offset_y = 4 if reg['id'] != 'alpha10' else -11
    offset_x = -36 if reg['id'] == 'iid' else 6
    ax_d.annotate(r'$\mu=0.001$', (drift_vals[1], f1_vals[1]),
                  textcoords='offset points', xytext=(offset_x, offset_y),
                  fontsize=8.0, color=reg['color'])

ax_d.set_xlabel(r'Parameter Drift ($L_2$ Norm) [Lower is More Stable]', fontweight='bold')
ax_d.set_ylabel(r'Global Macro-F1 Score (%) [Higher is Better]', fontweight='bold')
ax_d.set_title(r'(d) 2D Pareto Trade-off Plane', fontweight='bold', pad=10)
ax_d.set_xlim(-0.8, 24.5)
ax_d.set_ylim(41.0, 70.5)
ax_d.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
ax_d.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save to destination
output_path = 'thesis_report/figures/pareto_frontier_multiregime.png'
plt.savefig(output_path, dpi=300)
print(f"Successfully generated and saved minimalist figure to: {output_path}")

# Also update scratch copy
plt.savefig('scratch/generate_pareto_multiregime.png', dpi=300)
