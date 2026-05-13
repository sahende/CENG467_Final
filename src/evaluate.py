"""
CENG 467 - Model Evaluation and Comparison
Compares all models: Teacher, Student w/o Distillation, Distilled Student
"""

import torch
import json
import os
import time
import numpy as np
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import psutil

from config import Config
from models import get_teacher_model, get_student_model
from prepare_data import prepare_all_tasks

def get_memory_usage():
    """Returns current CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

def get_gpu_memory_usage():
    """Returns GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters())

def compute_all_metrics(predictions, labels):
    """Compute comprehensive metrics."""
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro', zero_division=0)
    rec = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

def evaluate_model(model, dataloader, device):
    """Evaluate model with all metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_all_metrics(all_preds, all_labels)
    metrics['avg_inference_time_ms'] = (sum(inference_times) / len(inference_times)) * 1000
    metrics['gpu_memory_mb'] = get_gpu_memory_usage()
    metrics['num_parameters'] = count_parameters(model)
    metrics['model_size_mb'] = metrics['num_parameters'] * 4 / (1024 * 1024)
    
    return metrics

def load_all_models(task, device):
    """Load all saved models for comparison."""
    models = {}
    
    # Teacher
    teacher_path = os.path.join(Config.MODEL_SAVE_PATH, f"teacher_{task}.pt")
    if os.path.exists(teacher_path):
        teacher = get_teacher_model(task)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher.to(device)
        teacher.eval()
        models['Teacher (BERT-base)'] = teacher
    
    # Student w/o distillation
    student_path = os.path.join(Config.MODEL_SAVE_PATH, f"student_no_distill_{task}.pt")
    if os.path.exists(student_path):
        student = get_student_model()
        student.load_state_dict(torch.load(student_path, map_location=device))
        student.to(device)
        student.eval()
        models['Student (w/o Distillation)'] = student
    
    # Distilled student
    distilled_path = os.path.join(Config.MODEL_SAVE_PATH, f"student_distilled_{task}.pt")
    if os.path.exists(distilled_path):
        distilled = get_student_model()
        distilled.load_state_dict(torch.load(distilled_path, map_location=device))
        distilled.to(device)
        distilled.eval()
        models['Student (Distilled)'] = distilled
    
    return models

def main():
    print("=" * 80)
    print("  CENG 467 - Model Evaluation & Comparison")
    print("  Knowledge Distillation for Task-Specific NLU")
    print("=" * 80)
    
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    task_dataloaders, _ = prepare_all_tasks()
    
    all_results = {}
    
    for task in Config.TASKS:
        print(f"\n{'='*60}")
        print(f"  Task: {task.upper()}")
        print(f"{'='*60}")
        
        models = load_all_models(task, device)
        test_loader = task_dataloaders[task]['test']
        
        task_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            metrics = evaluate_model(model, test_loader, device)
            task_results[model_name] = metrics
            
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  Recall:       {metrics['recall']:.4f}")
            print(f"  F1 Score:     {metrics['f1']:.4f}")
            print(f"  Parameters:   {metrics['num_parameters']/1e6:.1f}M")
            print(f"  Model Size:   {metrics['model_size_mb']:.1f} MB")
            print(f"  GPU Memory:   {metrics['gpu_memory_mb']:.1f} MB")
            print(f"  Inference:    {metrics['avg_inference_time_ms']:.1f} ms/batch")
        
        all_results[task] = task_results
        
        for model in models.values():
            del model
        torch.cuda.empty_cache()
    
    # Print summary table
    print(f"\n{'='*100}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*100}")
    
    table = []
    for task in Config.TASKS:
        for model_name, metrics in all_results[task].items():
            table.append([
                task.upper(),
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['num_parameters']/1e6:.1f}",
                f"{metrics['model_size_mb']:.1f}",
                f"{metrics['avg_inference_time_ms']:.1f}"
            ])
    
    headers = ['Task', 'Model', 'Acc', 'Prec', 'Rec', 'F1', 
               'Params(M)', 'Size(MB)', 'Time(ms)']
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Save results
    output_path = os.path.join(Config.RESULTS_PATH, "model_comparison.json")
    
    # Convert for JSON
    json_ready = {}
    for task in Config.TASKS:
        json_ready[task] = {}
        for model_name, metrics in all_results[task].items():
            json_ready[task][model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if k != 'confusion_matrix'
            }
            json_ready[task][model_name]['confusion_matrix'] = metrics['confusion_matrix']
    
    with open(output_path, 'w') as f:
        json.dump(json_ready, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()