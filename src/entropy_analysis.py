"""
Entropy & Calibration Analysis for Knowledge Distillation
Compares Teacher vs No Distill vs Direct KD vs HKD Students across ALL depths

FEATURES:
  ✅ All depths: 1L, 3L, 6L, 10L
  ✅ All datasets: RTE, MRPC 3.7K, MRPC 2.5K, CoLA 8.5K, CoLA 3.7K, CoLA 2.5K, SST-2 67K
  ✅ All models: Teacher, No Distill, Direct KD, HKD Students
  ✅ Metrics: Soft Entropy, ECE, Accuracy, MCC
  ✅ LaTeX table ready to copy-paste
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

from config import Config
from models import get_teacher_model, get_student_model
from prepare_data import prepare_all_tasks


# =========================
# CONSTANTS
# =========================
ALL_DEPTHS = [1, 3, 6, 10]

MODELS_DIR = Config.MODEL_SAVE_PATH
ALL_DATASET_DIR = os.path.join(MODELS_DIR, "all_dataset")
TEACHERS_DIR = os.path.join(MODELS_DIR, "teachers")
M2_MODELS_DIR = os.path.join(MODELS_DIR, "m2_models")


# =========================
# ASSISTANT/STUDENT MODEL LOADER
# =========================
def get_assistant_model(num_labels=2, num_layers=8):
    from transformers import BertConfig, AutoModelForSequenceClassification
    config = BertConfig.from_pretrained(
        Config.TEACHER_MODEL, num_labels=num_labels, num_hidden_layers=num_layers,
        output_hidden_states=False, output_attentions=False
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.init_weights()
    return model


# =========================
# METRICS
# =========================
def compute_softened_entropy(logits, temperature=Config.TEMPERATURE, eps=1e-12):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs + eps)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_confidence(logits):
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_calibration_metrics(confidences, accuracies, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_lower == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


# =========================
# ANALYZE ALL MODELS
# =========================
def analyze_models(teacher, models_dict, dataloader, device, task_name):
    """Analyze teacher + all student models."""
    teacher.eval()
    for model in models_dict.values():
        if model is not None:
            model.eval()
    
    all_teacher_entropy = []
    all_teacher_conf = []
    all_teacher_preds = []
    all_labels = []
    
    model_data = {name: {'entropy': [], 'conf': [], 'preds': []} 
                  for name in models_dict.keys()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {task_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]
            
            labels = batch["labels"]
            all_labels.extend(labels.cpu().numpy())
            
            # Teacher
            t_out = teacher(**inputs)
            all_teacher_entropy.append(compute_softened_entropy(t_out.logits))
            all_teacher_conf.append(compute_confidence(t_out.logits))
            all_teacher_preds.append(torch.argmax(t_out.logits, dim=-1))
            
            # Other models
            for name, model in models_dict.items():
                if model is None:
                    continue
                s_out = model(**inputs)
                model_data[name]['entropy'].append(compute_softened_entropy(s_out.logits))
                model_data[name]['conf'].append(compute_confidence(s_out.logits))
                model_data[name]['preds'].append(torch.argmax(s_out.logits, dim=-1))
    
    all_labels_tensor = torch.tensor(all_labels, device=device)
    
    # Teacher summary
    t_entropy = torch.cat(all_teacher_entropy)
    t_conf = torch.cat(all_teacher_conf)
    t_preds = torch.cat(all_teacher_preds)
    t_acc = (t_preds == all_labels_tensor).float()
    t_mcc = matthews_corrcoef(all_labels_tensor.cpu().numpy(), t_preds.cpu().numpy())
    
    teacher_summary = {
        'mean_softened_entropy': float(t_entropy.mean()),
        'accuracy': float(t_acc.mean()),
        'mcc': float(t_mcc),
        'calibration': {'ece': compute_calibration_metrics(t_conf.cpu().numpy(), t_acc.cpu().numpy())}
    }
    
    # Student summaries
    student_summaries = {}
    for name, data in model_data.items():
        if not data['entropy']:
            continue
        
        entropy = torch.cat(data['entropy'])
        conf = torch.cat(data['conf'])
        preds = torch.cat(data['preds'])
        acc = (preds == all_labels_tensor).float()
        mcc = matthews_corrcoef(all_labels_tensor.cpu().numpy(), preds.cpu().numpy())
        
        student_summaries[name] = {
            'mean_softened_entropy': float(entropy.mean()),
            'accuracy': float(acc.mean()),
            'mcc': float(mcc),
            'calibration': {'ece': compute_calibration_metrics(conf.cpu().numpy(), acc.cpu().numpy())}
        }
    
    return {'teacher': teacher_summary, 'students': student_summaries}


# =========================
# MODEL LOADING HELPERS
# =========================
def find_file(paths):
    """Find first existing file from a list of possible paths."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_model_path(task_name, model_type, depth, suffix):
    """
    Find correct checkpoint for subsampled variants.
    Searches 4 locations in priority order (same as CKA analysis).
    """
    # 1. models/{task}/{task}_{type}_{depth}L{suffix}.pt
    path1 = f"models/{task_name}/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(path1):
        return path1
    
    # 2. models/all_dataset/{task}_{type}_{depth}L{suffix}.pt
    path2 = f"models/all_dataset/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(path2):
        return path2
    
    # 3. models/all_dataset/{task}_{type}_{depth}L.pt (suffix'siz, original)
    path3 = f"models/all_dataset/{task_name}_{model_type}_{depth}L.pt"
    if os.path.exists(path3):
        return path3
    
    # 4. models/{task}_{type}_{depth}L{suffix}.pt
    path4 = f"models/{task_name}_{model_type}_{depth}L{suffix}.pt"
    if os.path.exists(path4):
        return path4
    
    return None


def load_teacher(task_name, suffix, device):
    """Load teacher model."""
    teacher = get_teacher_model(task_name, num_labels=2)
    path = find_file([
        os.path.join(TEACHERS_DIR, f"teacher_{task_name}{suffix}.pt"),
        os.path.join(MODELS_DIR, f"teacher_{task_name}.pt"),
    ])
    if path:
        teacher.load_state_dict(torch.load(path, map_location=device))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"    ✓ Teacher: {os.path.basename(path)}")
        return teacher
    print(f"    ✗ Teacher not found (suffix={suffix})")
    return None


def load_all_models(task_name, suffix, device):
    """
    Load all student models for ALL depths.
    Uses find_model_path for HKD students (same logic as CKA analysis).
    """
    models = {}
    
    # HKD Students for all depths (using same path logic as CKA)
    for depth in ALL_DEPTHS:
        hkd_path = find_model_path(task_name, "student", depth, suffix)
        if hkd_path:
            student = get_student_model(num_labels=2)
            student.load_state_dict(torch.load(hkd_path, map_location=device))
            student.to(device).eval()
            models[f'HKD {depth}L ($\\rightarrow$6L)'] = student
            print(f"    ✓ HKD {depth}L Student: {os.path.basename(hkd_path)}")
        else:
            print(f"    ✗ HKD {depth}L Student NOT FOUND (task={task_name}, suffix={suffix})")
    
    # No Distill (6L, no KD)
    no_distill_path = find_file([
        os.path.join(MODELS_DIR, f"student_no_distill_{task_name}{suffix}.pt"),
        os.path.join(MODELS_DIR, f"student_no_distill_{task_name}.pt"),
    ])
    if no_distill_path:
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(no_distill_path, map_location=device))
        student.to(device).eval()
        models['No Distill (6L)'] = student
        print(f"    ✓ No Distill: {os.path.basename(no_distill_path)}")
    else:
        print(f"    ✗ No Distill NOT FOUND (task={task_name}, suffix={suffix})")
    
    # Direct KD (12L→6L)
    direct_kd_path = find_file([
        os.path.join(MODELS_DIR, f"student_kd_{task_name}{suffix}.pt"),
        os.path.join(MODELS_DIR, f"student_kd_{task_name}.pt"),
        os.path.join(MODELS_DIR, f"student_distilled_{task_name}.pt"),
    ])
    if direct_kd_path:
        student = get_student_model(num_labels=2)
        student.load_state_dict(torch.load(direct_kd_path, map_location=device))
        student.to(device).eval()
        models['Direct KD (6L)'] = student
        print(f"    ✓ Direct KD: {os.path.basename(direct_kd_path)}")
    else:
        print(f"    ✗ Direct KD NOT FOUND (task={task_name}, suffix={suffix})")
    
    return models


def run_config(task_name, variant_label, suffix, task_subsample_sizes, device, all_results):
    """Run analysis for one configuration."""
    print(f"\n{'─'*60}")
    print(f"  {task_name.upper()} - {variant_label}")
    print(f"{'─'*60}")
    
    target_data, _ = prepare_all_tasks([task_name], task_subsample_sizes)
    test_loader = target_data[task_name]['test']
    
    teacher = load_teacher(task_name, suffix, device)
    if teacher is None:
        return
    
    models = load_all_models(task_name, suffix, device)
    if not models:
        print(f"    ✗ No models found")
        return
    
    summaries = analyze_models(teacher, models, test_loader, device, task_name)
    
    key = f"{task_name}_{variant_label}"
    all_results[key] = summaries
    
    # Print summary
    t = summaries['teacher']
    print(f"    {'Teacher':<25} Acc={t['accuracy']:.4f} MCC={t['mcc']:.4f} "
          f"Entropy={t['mean_softened_entropy']:.4f} ECE={t['calibration']['ece']:.4f}")
    
    for name, s in summaries['students'].items():
        mcc_str = f"MCC={s['mcc']:.4f}" if s.get('mcc') is not None else ""
        print(f"    {name:<25} Acc={s['accuracy']:.4f} {mcc_str} "
              f"Entropy={s['mean_softened_entropy']:.4f} ECE={s['calibration']['ece']:.4f}")
    
    del teacher
    torch.cuda.empty_cache()


def main():
    device = Config.DEVICE
    
    print("\n" + "=" * 70)
    print("  ENTROPY & CALIBRATION ANALYSIS (All Depths)")
    print("=" * 70)
    
    all_results = {}
    
    # ================================================================
    # ALL CONFIGURATIONS
    # ================================================================
    configs = [
        ("cola", "8.5K", "_original", {}),
        ("cola", "3.7K", "_3668", {"cola": 3668}),
        ("cola", "2.5K", "_2490", {"cola": 2490}),
        ("mrpc", "3.7K", "_original", {}),
        ("mrpc", "2.5K", "_2490", {"mrpc": 2490}),
        ("rte", "2.5K", "_original", {}),
        ("sst2", "67K", "_original", {}),
    ]
    
    for task, variant, suffix, subsample in configs:
        run_config(task, variant, suffix, subsample, device, all_results)
    
    # ================================================================
    # SAVE JSON
    # ================================================================
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_to_serializable(v) for v in obj]
        return obj
    
    output_path = os.path.join(Config.RESULTS_PATH, "entropy_analysis_all_depths.json")
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # ================================================================
    # LATEX TABLE - ALL DEPTHS
    # ================================================================
    print(f"\n{'='*90}")
    print("  LATEX TABLE (Copy-Paste Ready)")
    print(f"{'='*90}")
    
    task_order = [
        ('cola', '8.5K', 'CoLA 8.5K'),
        ('cola', '3.7K', 'CoLA 3.7K'),
        ('cola', '2.5K', 'CoLA 2.5K'),
        ('mrpc', '3.7K', 'MRPC 3.7K'),
        ('mrpc', '2.5K', 'MRPC 2.5K'),
        ('rte', '2.5K', 'RTE 2.5K'),
        ('sst2', '67K', 'SST-2 67K'),
    ]
    
    hkd_models = [f'HKD {d}L ($\\rightarrow$6L)' for d in ALL_DEPTHS]
    baseline_models = ['No Distill (6L)', 'Direct KD (6L)']
    
    for task, variant, label in task_order:
        key = f"{task}_{variant}"
        if key not in all_results:
            continue
        
        r = all_results[key]
        
        print(f"\n\\multicolumn{{5}}{{|c|}}{{\\textbf{{{label}}}}} \\\\")
        print("\\hline")
        
        # Teacher
        t = r['teacher']
        print(f"Teacher (12L) & {t['mean_softened_entropy']:.3f} & {t['calibration']['ece']:.3f} & {t['accuracy']:.3f} & {t['mcc']:.3f} \\\\")
        
        # Baseline models
        for model_name in baseline_models:
            s = r['students'].get(model_name)
            if s:
                mcc_str = f"{s['mcc']:.3f}" if s.get('mcc') is not None else "N/A"
                print(f"{model_name} & {s['mean_softened_entropy']:.3f} & {s['calibration']['ece']:.3f} & {s['accuracy']:.3f} & {mcc_str} \\\\")
        
        # HKD models
        for model_name in hkd_models:
            s = r['students'].get(model_name)
            if s:
                mcc_str = f"{s['mcc']:.3f}" if s.get('mcc') is not None else "N/A"
                print(f"{model_name} & {s['mean_softened_entropy']:.3f} & {s['calibration']['ece']:.3f} & {s['accuracy']:.3f} & {mcc_str} \\\\")
        
        print("\\hline")
    
    print(f"\n✓ JSON saved to: {output_path}")


if __name__ == "__main__":
    main()