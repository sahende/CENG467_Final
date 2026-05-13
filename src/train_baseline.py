"""
CENG 467 - Baseline Model Training
Trains teacher (BERT-base) and student (w/o distillation) as baselines.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import os
import time
import random

from config import Config
from models import get_teacher_model, get_student_model, count_parameters
from prepare_data import prepare_all_tasks

# Reproducibility settings
torch.manual_seed(Config.SEED)
torch.cuda.manual_seed_all(Config.SEED)
np.random.seed(Config.SEED)
random.seed(Config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_all_metrics(predictions, labels):
    """Compute comprehensive metrics."""
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro', zero_division=0)
    rec = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    prec_per_class = precision_score(labels, predictions, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(labels, predictions, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0).tolist()
    
    return {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
        'precision_per_class': prec_per_class,
        'recall_per_class': rec_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm.tolist()
    }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(all_preds, all_labels)
    
    return avg_loss, metrics

def evaluate(model, dataloader, device):
    """Evaluate the model with comprehensive metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    inference_times = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Measure inference time
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
            
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(all_preds, all_labels)
    metrics['avg_inference_time_ms'] = (sum(inference_times) / len(inference_times)) * 1000
    metrics['total_inference_time_s'] = sum(inference_times)
    
    return avg_loss, metrics

def train_model(model, model_name, task_name, train_loader, val_loader, test_loader, device):
    """Complete training loop with early stopping."""
    print(f"\n{'='*50}")
    print(f"Training {model_name} on {task_name.upper()}")
    print(f"{'='*50}")
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    warmup_steps = int(Config.WARMUP_RATIO * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    results = {'train': [], 'val': [], 'test': None}
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        results['train'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            **train_metrics
        })
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        
        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, device)
        results['val'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            **val_metrics
        })
        
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")
        
        # Early stopping check
        if val_metrics['f1_macro'] > best_val_f1 + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✓ New best model! F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break
    
    # Load best model for test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    print(f"\n{'='*50}")
    print(f"Final Test Evaluation (using best model from epoch {best_epoch})")
    print(f"{'='*50}")
    
    test_loss, test_metrics = evaluate(model, test_loader, device)
    test_metrics['num_parameters'] = count_parameters(model)
    test_metrics['best_epoch'] = best_epoch
    results['test'] = {'loss': test_loss, **test_metrics}
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_macro']:.4f}")
    print(f"  Avg Inference: {test_metrics['avg_inference_time_ms']:.2f} ms/batch")
    print(f"  Parameters: {test_metrics['num_parameters']:,}")
    
    return model, results

def main():
    """Main function to train baseline models."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CENG 467 - Baseline Model Training                    ║")
    print("║  Knowledge Distillation for Task-Specific NLU          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    task_dataloaders, tokenizer = prepare_all_tasks()
    
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    all_results = {}
    
    for task in Config.TASKS:
        task_results = {}
        
        train_loader = task_dataloaders[task]['train']
        val_loader = task_dataloaders[task]['val']
        test_loader = task_dataloaders[task]['test']
        
        # ============ TEACHER ============
        print(f"\n{'#'*55}")
        print(f"  BASELINE 1: Teacher Model (BERT-base) - {task.upper()}")
        print(f"{'#'*55}")
        
        teacher = get_teacher_model(task)
        teacher.to(device)
        
        teacher, teacher_results = train_model(
            teacher, "BERT-base Teacher", task,
            train_loader, val_loader, test_loader, device
        )
        
        torch.save(
            teacher.state_dict(),
            os.path.join(Config.MODEL_SAVE_PATH, f"teacher_{task}.pt")
        )
        task_results['teacher'] = teacher_results
        
        # ============ STUDENT w/o DISTILLATION ============
        print(f"\n{'#'*55}")
        print(f"  BASELINE 2: Student Model (w/o Distillation) - {task.upper()}")
        print(f"{'#'*55}")
        
        student_no_distill = get_student_model()
        student_no_distill.to(device)
        
        student_no_distill, student_results = train_model(
            student_no_distill, "Student w/o Distillation", task,
            train_loader, val_loader, test_loader, device
        )
        
        torch.save(
            student_no_distill.state_dict(),
            os.path.join(Config.MODEL_SAVE_PATH, f"student_no_distill_{task}.pt")
        )
        task_results['student_no_distill'] = student_results
        
        all_results[task] = task_results
        
        del teacher, student_no_distill
        torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(Config.RESULTS_PATH, "baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary table
    print(f"\n{'='*80}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Task':<6} {'Model':<28} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Params':<10} {'Time(ms)':<10}")
    print(f"{'-'*80}")
    
    for task in Config.TASKS:
        t = all_results[task]['teacher']['test']
        s = all_results[task]['student_no_distill']['test']
        
        print(f"{task.upper():<6} {'Teacher (BERT-base)':<28} "
              f"{t['accuracy']:.4f}   {t['precision_macro']:.4f}   {t['recall_macro']:.4f}   "
              f"{t['f1_macro']:.4f}   {t['num_parameters']/1e6:.1f}M      {t['avg_inference_time_ms']:.1f}")
        print(f"{'':<6} {'Student (w/o Distillation)':<28} "
              f"{s['accuracy']:.4f}   {s['precision_macro']:.4f}   {s['recall_macro']:.4f}   "
              f"{s['f1_macro']:.4f}   {s['num_parameters']/1e6:.1f}M      {s['avg_inference_time_ms']:.1f}")
        print(f"{'-'*80}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()