"""
CKA / SVCCA Analysis for Hierarchical Knowledge Distillation
Analyzes representation similarity across tasks, depths, and data scales.

FIXED:
  ✅ Standardization REMOVED (Linear CKA is scale-invariant)
  ✅ PWCCA REMOVED (non-standard implementation)
  ✅ SVCCA renamed to "SVCCA-inspired"
  ✅ CKA renamed to "SVD-denoised CKA"
  ✅ n_components clipping FIXED: max(5, n) → min(max(5, n), len(S))
  ✅ MRPC 2.5K checkpoint paths FIXED (now uses MRPC-specific checkpoints)
  ✅ Numerically stable HSIC computation
  ✅ Teacher path with suffix support
  ✅ Auto path detection for subsampled variants
"""

import torch
import numpy as np
import os
import json
from tqdm import tqdm

from config import Config
from models import get_teacher_model, get_student_model
from hierarchical_knowledge_distillation import get_assistant_model, evaluate
from prepare_data import prepare_all_tasks


# =========================
# SVD-DENOISED CKA
# =========================
def compute_cka(X, Y, variance_threshold=0.99):
    """
    SVD-denoised variant of Linear CKA (Kornblith et al., 2019).
    First applies SVD-based dimensionality reduction retaining `variance_threshold`
    of variance, then computes HSIC on the cleaned representations.
    Linear CKA is scale-invariant, so no standardization is applied.
    
    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    # Center the columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # SVD denoising for X
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    explained_var_x = np.cumsum(Sx**2) / np.sum(Sx**2)
    n_components_x = np.searchsorted(explained_var_x, variance_threshold) + 1
    # FIX: Clip safely (min(max(5, n), len(S)))
    n_components_x = min(max(5, n_components_x), len(Sx))
    
    # SVD denoising for Y
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    explained_var_y = np.cumsum(Sy**2) / np.sum(Sy**2)
    n_components_y = np.searchsorted(explained_var_y, variance_threshold) + 1
    # FIX: Clip safely (min(max(5, n), len(S)))
    n_components_y = min(max(5, n_components_y), len(Sy))
    
    # Use the SAME number of components for both
    n_components = min(n_components_x, n_components_y)
    
    X_clean = Ux[:, :n_components] @ np.diag(Sx[:n_components])
    Y_clean = Uy[:, :n_components] @ np.diag(Sy[:n_components])
    
    # Linear CKA on cleaned representations
    n = X_clean.shape[0]
    
    # Gram matrices
    K_XX = X_clean @ X_clean.T
    K_YY = Y_clean @ Y_clean.T
    K_XY = X_clean @ Y_clean.T
    
    # HSIC via Frobenius norm squared (numerically stable)
    hsic_XY = np.sum(K_XY * K_XY) / (n - 1)**2
    hsic_XX = np.sum(K_XX * K_XX) / (n - 1)**2
    hsic_YY = np.sum(K_YY * K_YY) / (n - 1)**2
    
    denom = np.sqrt(max(0.0, hsic_XX * hsic_YY))
    if denom < 1e-10:
        return 0.0
    
    cka = hsic_XY / denom
    return float(max(0.0, min(1.0, cka)))


# =========================
# SVCCA-INSPIRED
# =========================
def compute_svcca(X, Y, variance_threshold=0.99):
    """
    SVCCA-inspired approach:
    SVD-based dimensionality reduction followed by CCA on the reduced representations.
    Uses U_k @ diag(S_k) projection.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    explained_var_x = np.cumsum(Sx**2) / np.sum(Sx**2)
    n_components_x = np.searchsorted(explained_var_x, variance_threshold) + 1
    n_components_x = min(n_components_x, len(Sx))
    X_reduced = Ux[:, :n_components_x] @ np.diag(Sx[:n_components_x])
    
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    explained_var_y = np.cumsum(Sy**2) / np.sum(Sy**2)
    n_components_y = np.searchsorted(explained_var_y, variance_threshold) + 1
    n_components_y = min(n_components_y, len(Sy))
    Y_reduced = Uy[:, :n_components_y] @ np.diag(Sy[:n_components_y])
    
    return compute_cca(X_reduced, Y_reduced)


def compute_cca(X, Y):
    """Canonical Correlation Analysis: mean of top 10 canonical correlations."""
    n = X.shape[0]
    Sigma_XX = (X.T @ X) / (n - 1) + 1e-5 * np.eye(X.shape[1])
    Sigma_YY = (Y.T @ Y) / (n - 1) + 1e-5 * np.eye(Y.shape[1])
    Sigma_XY = (X.T @ Y) / (n - 1)
    try:
        inv_sqrt_XX = np.linalg.inv(np.linalg.cholesky(Sigma_XX))
        inv_sqrt_YY = np.linalg.inv(np.linalg.cholesky(Sigma_YY))
        M = inv_sqrt_XX @ Sigma_XY @ inv_sqrt_YY.T
        _, S, _ = np.linalg.svd(M)
        top_k = min(10, len(S))
        return float(np.mean(S[:top_k]))
    except np.linalg.LinAlgError:
        return 0.0


# =========================
# EMBEDDING EXTRACTION
# =========================
def get_cls_embeddings(model, loader, device, max_samples=500):
    """Extract CLS token embeddings from the final hidden layer."""
    model.eval()
    all_hidden = []
    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Embeddings"):
            if max_samples and n_samples >= max_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            outputs = model(**inputs, output_hidden_states=True)
            cls_hidden = outputs.hidden_states[-1][:, 0, :]
            all_hidden.append(cls_hidden.cpu().numpy())
            n_samples += len(cls_hidden)
    result = np.concatenate(all_hidden, axis=0)
    if max_samples and len(result) > max_samples:
        result = result[:max_samples]
    return result


# =========================
# MODEL LOADING
# =========================
def load_model_from_path(model, path, device):
    if os.path.exists(path):
        print(f"    ✓ Found: {os.path.basename(path)}")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, True
    else:
        print(f"    ✗ NOT FOUND: {path}")
        return model, False


# =========================
# PATH HELPER: Find correct checkpoint for subsampled variants
# =========================
def find_model_path(task_name, model_type, depth, suffix):
    # 1. Task-specific path (for subsampled variants)
    task_specific = f"models/{task_name}/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(task_specific):
        return task_specific
    
    # 2. Shared path WITH suffix (for subsampled variants in all_dataset)
    shared_with_suffix = f"models/all_dataset/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(shared_with_suffix):
        return shared_with_suffix
    
    # 3. Shared path WITHOUT suffix (for original datasets)
    shared = f"models/all_dataset/{task_name}_{model_type}_{depth}L.pt"
    if os.path.exists(shared):
        return shared
    
    # 4. Fallback
    fallback = f"models/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(fallback):
        return fallback
    
    return None


# =========================
# SINGLE CONFIG ANALYSIS
# =========================
def analyze_config(task_name, config_label, device, 
                   test_loader, teacher_suffix="_original"):
    """
    Analyze one configuration (task + data scale).
    Automatically finds correct model paths based on suffix.
    """
    print(f"\n{'─'*60}")
    print(f"  {config_label}")
    print(f"{'─'*60}")
    
    # Teacher with suffix
    teacher = get_teacher_model(task_name, num_labels=2)
    teacher_path = f"models/teachers/teacher_{task_name}{teacher_suffix}.pt"
    teacher, ok = load_model_from_path(teacher, teacher_path, device)
    if not ok:
        return None
    teacher_emb = get_cls_embeddings(teacher, test_loader, device)
    print(f"  Teacher embeddings: {teacher_emb.shape}")
    
    depths = [1, 3, 6, 10]
    results = {
        'cka': {'T-A': {}, 'T-S': {}, 'A-S': {}},
        'svcca': {'T-A': {}, 'T-S': {}, 'A-S': {}},
        'f1_assistant': {},
        'f1_student': {},
    }
    
    for depth in depths:
        print(f"\n  --- Depth {depth}L ---")
        
        # Find correct model paths
        assistant_path = find_model_path(task_name, "assistant", depth, teacher_suffix)
        student_path = find_model_path(task_name, "student", depth, teacher_suffix)
        
        if not assistant_path or not student_path:
            print(f"    ✗ Models not found (suffix={teacher_suffix})")
            continue
        
        # Assistant
        assistant = get_assistant_model(num_labels=2, num_layers=depth)
        assistant, ok_a = load_model_from_path(assistant, assistant_path, device)
        if not ok_a:
            del assistant
            continue
        
        # Student
        student = get_student_model(num_labels=2)
        student, ok_s = load_model_from_path(student, student_path, device)
        if not ok_s:
            del assistant, student
            continue
        
        # F1 Evaluation
        _, assistant_metrics = evaluate(assistant, test_loader, device)
        _, student_metrics = evaluate(student, test_loader, device)
        results['f1_assistant'][str(depth)] = round(assistant_metrics['f1_macro'], 4)
        results['f1_student'][str(depth)] = round(student_metrics['f1_macro'], 4)
        
        # Embeddings
        assistant_emb = get_cls_embeddings(assistant, test_loader, device)
        student_emb = get_cls_embeddings(student, test_loader, device)
        
        # SVD-Denoised CKA
        results['cka']['T-A'][str(depth)] = round(compute_cka(teacher_emb, assistant_emb), 4)
        results['cka']['T-S'][str(depth)] = round(compute_cka(teacher_emb, student_emb), 4)
        results['cka']['A-S'][str(depth)] = round(compute_cka(assistant_emb, student_emb), 4)
        
        # SVCCA-inspired
        results['svcca']['T-A'][str(depth)] = round(compute_svcca(teacher_emb, assistant_emb), 4)
        results['svcca']['T-S'][str(depth)] = round(compute_svcca(teacher_emb, student_emb), 4)
        results['svcca']['A-S'][str(depth)] = round(compute_svcca(assistant_emb, student_emb), 4)
        
        print(f"    F1: Assistant={assistant_metrics['f1_macro']:.4f}, Student={student_metrics['f1_macro']:.4f}")
        print(f"    CKA T-A={results['cka']['T-A'][str(depth)]:.4f}, "
              f"SVCCA T-A={results['svcca']['T-A'][str(depth)]:.4f}")
        
        del assistant, student
        torch.cuda.empty_cache()
    
    del teacher
    torch.cuda.empty_cache()
    
    return results


# =========================
# MAIN
# =========================
def main():
    device = Config.DEVICE
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    all_results = {}
    
    # ================================================================
    # ALL CONFIGURATIONS
    # ================================================================
    configs = [
        # (task, config_label, task_list, subsample_sizes, teacher_suffix)
        ("cola", "CoLA 8.5K", ["cola"], {}, "_original"),
        ("cola", "CoLA 3.7K", ["cola"], {"cola": 3668}, "_3668"),
        ("cola", "CoLA 2.5K", ["cola"], {"cola": 2490}, "_2490"),
        ("mrpc", "MRPC 3.7K", ["mrpc"], {}, "_original"),
        ("mrpc", "MRPC 2.5K", ["mrpc"], {"mrpc": 2490}, "_2490"),
        ("rte", "RTE 2.5K", ["rte"], {}, "_original"),
        ("sst2", "SST-2 67K", ["sst2"], {}, "_original"),
    ]
    
    for task, label, task_list, subsample, suffix in configs:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        
        target_data, _ = prepare_all_tasks(task_list, subsample)
        test_loader = target_data[task]['test']
        
        results = analyze_config(task, label, device, test_loader, teacher_suffix=suffix)
        if results:
            key = label.lower().replace(" ", "_").replace(".", "")
            all_results[key] = results
    
    # ================================================================
    # SAVE
    # ================================================================
    def make_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [make_serializable(v) for v in obj]
        return obj
    
    with open(os.path.join(Config.RESULTS_PATH, "cka_svcca_results.json"), 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    
    # ================================================================
    # SUMMARY TABLE: CKA + F1
    # ================================================================
    print(f"\n{'='*120}")
    print("  SVD-Denoised CKA: Teacher vs Assistant (with Student F1)")
    print(f"{'='*120}")
    header = f"  {'Config':<22}"
    for d in [1, 3, 6, 10]:
        header += f" {'CKA'+str(d)+'L':<10} {'F1'+str(d)+'L':<10}"
    print(header)
    print(f"  {'─'*105}")
    
    for config, results in all_results.items():
        if results and 'cka' in results:
            parts = [config]
            for d in [1, 3, 6, 10]:
                cka = results['cka']['T-A'].get(str(d), 'N/A')
                f1 = results.get('f1_student', {}).get(str(d), 'N/A')
                parts.append(f"{cka:.4f}" if isinstance(cka, float) else str(cka))
                parts.append(f"{f1:.4f}" if isinstance(f1, float) else str(f1))
            print(f"  {parts[0]:<22} {parts[1]:<10} {parts[2]:<10} {parts[3]:<10} {parts[4]:<10} {parts[5]:<10} {parts[6]:<10} {parts[7]:<10} {parts[8]:<10}")
    
    # ================================================================
    # SUMMARY TABLE: SVCCA
    # ================================================================
    print(f"\n{'='*120}")
    print("  SVCCA-inspired: Teacher vs Assistant")
    print(f"{'='*120}")
    header = f"  {'Config':<22}"
    for d in [1, 3, 6, 10]:
        header += f" {'SV'+str(d)+'L':<10}"
    print(header)
    print(f"  {'─'*65}")
    
    for config, results in all_results.items():
        if results and 'svcca' in results:
            parts = [config]
            for d in [1, 3, 6, 10]:
                sv = results['svcca']['T-A'].get(str(d), 'N/A')
                parts.append(f"{sv:.4f}" if isinstance(sv, float) else str(sv))
            print(f"  {parts[0]:<22} {parts[1]:<10} {parts[2]:<10} {parts[3]:<10} {parts[4]:<10}")


if __name__ == "__main__":
    main()