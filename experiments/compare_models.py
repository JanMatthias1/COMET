import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# --- DIRECTORIES ---
# 1. Baseline COMET Run (From your very first successful test)
baseline_dir = "in_house_lnp_library_infer_results_withclsrep_OS/demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dDeployTrain_allastest_infer_results/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V0_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19"

# 2. Your New GIN Fusion (Strategy B) Run
gin_dir = "./infer_results/infer_GIN_StrategyB_Run"

def get_data_from_dir(directory):
    """Finds the .out.pkl file in a directory and dynamically extracts targets/predictions."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
        
    pkl_file = next((f for f in os.listdir(directory) if f.endswith(".out.pkl")), None)
    if not pkl_file:
        print(f"No .out.pkl file found in: {directory}")
        return None
        
    with open(os.path.join(directory, pkl_file), "rb") as f:
        data = pickle.load(f)
        
    # COMET's output keys change based on the --valid-subset flag (test vs infer)
    suffix = 'test' if any('test_target' in k for k in data.keys()) else 'infer'
    
    return {
        'b16_targ': np.array(data[f'in_house_lnp_B16F10_luc{suffix}_target']).flatten(),
        'b16_pred': np.array(data[f'in_house_lnp_B16F10_luc{suffix}_predict']).flatten(),
        'dc24_targ': np.array(data[f'in_house_lnp_DC24_luc{suffix}_target']).flatten(),
        'dc24_pred': np.array(data[f'in_house_lnp_DC24_luc{suffix}_predict']).flatten()
    }

print("Loading Baseline Data...")
baseline_data = get_data_from_dir(baseline_dir)

print("Loading New GIN Data...")
gin_data = get_data_from_dir(gin_dir)

if baseline_data and gin_data:
    
    # --- PRINT METRICS TO TERMINAL ---
    print("\n" + "="*50)
    print("📊 METRICS SUMMARY")
    print("="*50)
    
    datasets = [
        (baseline_data, "Baseline COMET"),
        (gin_data, "GIN Strategy B (Gated Residual)")
    ]
    
    for data, model_name in datasets:
        print(f"\n--- {model_name} ---")
        for prefix, display_name in [('b16', 'B16F10'), ('dc24', 'DC2.4')]:
            targ = data[f'{prefix}_targ']
            pred = data[f'{prefix}_pred']
            
            spearman_corr, _ = spearmanr(pred, targ)
            pearson_corr, _ = pearsonr(pred, targ)
            
            print(f"{display_name} Cells:")
            print(f"  • Spearman R: {spearman_corr:.4f}")
            print(f"  • Pearson R:  {pearson_corr:.4f}")
            
    print("="*50 + "\n")

    # --- SETUP AND SAVE PLOT ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Baseline COMET vs. GIN Strategy B", fontsize=16, fontweight='bold')
    
    for row_idx, (data, model_name) in enumerate(datasets):
        for col_idx, cell_line in enumerate([('b16', 'B16F10', 'blue'), ('dc24', 'DC2.4', 'green')]):
            prefix, display_name, color = cell_line
            targ = data[f'{prefix}_targ']
            pred = data[f'{prefix}_pred']
            
            corr, _ = spearmanr(pred, targ)
            
            ax = axes[row_idx][col_idx]
            ax.scatter(targ, pred, alpha=0.5, color=color, edgecolor='k')
            ax.set_title(f"[{model_name}]\n{display_name} Cells (Spearman R = {corr:.3f})", fontsize=12)
            ax.set_xlabel("True Target Value")
            ax.set_ylabel("Predicted Value")
            
            min_v = min(min(targ), min(pred))
            max_v = max(max(targ), max(pred))
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Ideal (y=x)')
            ax.legend()
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
    save_path = "model_comparison_scatter.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Comparison plot saved successfully to: {os.path.abspath(save_path)}")
else:
    print("\n❌ Could not generate plot. One or both datasets failed to load.")