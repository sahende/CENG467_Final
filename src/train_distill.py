"""
CENG 467 - Knowledge Distillation Training
Trains the student model using knowledge distillation from the teacher.
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
from models import (
    get_teacher_model, get_student_model, 
    DistillationLoss, count_parameters
)
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

def train_distill_epoch(teacher, student, dataloader, optimizer, 
                        scheduler, distillation_loss_fn, device):
    """Train student model for one epoch using distillation."""
    teacher.eval()
    student.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Distilling")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            teacher_logits = teacher_outputs.logits
        
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        student_logits = student_outputs.logits
        
        loss, ce_loss, kl_loss = distillation_loss_fn(
            student_logits, teacher_logits, labels
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_kl_loss += kl_loss.item()
        
        predictions = torch.argmax(student_logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'KL': f'{kl_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)
    metrics = compute_all_metrics(all_preds, all_labels)
    metrics['ce_loss'] = avg_ce
    metrics['kl_loss'] = avg_kl
    
    return avg_loss, metrics

def evaluate(model, dataloader, device):
    """Evaluate the model."""
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
    
    return avg_loss, metrics

def train_distilled_model(teacher, student, task_name, train_loader, 
                         val_loader, test_loader, device):
    """Train student model using knowledge distillation with early stopping."""
    print(f"\n{'='*50}")
    print(f"Knowledge Distillation Training on {task_name.upper()}")
    print(f"{'='*50}")
    
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    student.to(device)
    
    distillation_loss_fn = DistillationLoss(
        temperature=Config.TEMPERATURE,
        alpha=Config.ALPHA
    )
    
    optimizer = AdamW(
        student.parameters(),
        lr=Config.STUDENT_LR,
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
        
        train_loss, train_metrics = train_distill_epoch(
            teacher, student, train_loader, optimizer, 
            scheduler, distillation_loss_fn, device
        )
        
        results['train'].append({
            'epoch': epoch + 1,
            'loss': train_loss,
            **train_metrics
        })
        
        print(f"Train Loss: {train_loss:.4f} (CE: {train_metrics['ce_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.4f})")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        
        val_loss, val_metrics = evaluate(student, val_loader, device)
        results['val'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            **val_metrics
        })
        
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")
        
        if val_metrics['f1_macro'] > best_val_f1 + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            print(f"✓ New best student! F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
                break
    
    if best_model_state is not None:
        student.load_state_dict(best_model_state)
    
    print(f"\n{'='*50}")
    print(f"Final Test Evaluation (using best model from epoch {best_epoch})")
    print(f"{'='*50}")
    
    test_loss, test_metrics = evaluate(student, test_loader, device)
    test_metrics['num_parameters'] = count_parameters(student)
    test_metrics['best_epoch'] = best_epoch
    results['test'] = {'loss': test_loss, **test_metrics}
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_macro']:.4f}")
    print(f"  Avg Inference: {test_metrics['avg_inference_time_ms']:.2f} ms/batch")
    print(f"  Parameters: {test_metrics['num_parameters']:,}")
    
    return student, results

def main():
    """Main function for knowledge distillation training."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CENG 467 - Knowledge Distillation Training             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    task_dataloaders, tokenizer = prepare_all_tasks()
    
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    all_results = {}
    
    for task in Config.TASKS:
        print(f"\n{'#'*55}")
        print(f"  Knowledge Distillation - {task.upper()}")
        print(f"{'#'*55}")
        
        teacher_path = os.path.join(Config.MODEL_SAVE_PATH, f"teacher_{task}.pt")
        
        if os.path.exists(teacher_path):
            print(f"Loading saved teacher model...")
            teacher = get_teacher_model(task)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        else:
            print("No saved teacher found! Run train_baseline.py first.")
            continue
        
        student = get_student_model()
        
        train_loader = task_dataloaders[task]['train']
        val_loader = task_dataloaders[task]['val']
        test_loader = task_dataloaders[task]['test']
        
        distilled_student, results = train_distilled_model(
            teacher, student, task,
            train_loader, val_loader, test_loader, device
        )
        
        torch.save(
            distilled_student.state_dict(),
            os.path.join(Config.MODEL_SAVE_PATH, f"student_distilled_{task}.pt")
        )
        
        all_results[task] = results
        
        del teacher, student, distilled_student
        torch.cuda.empty_cache()
    
    results_file = os.path.join(Config.RESULTS_PATH, "distillation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("  DISTILLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for task in Config.TASKS:
        t = all_results[task]['test']
        print(f"\n{task.upper()}:")
        print(f"  Accuracy:  {t['accuracy']:.4f}")
        print(f"  Precision: {t['precision_macro']:.4f}")
        print(f"  Recall:    {t['recall_macro']:.4f}")
        print(f"  F1 Score:  {t['f1_macro']:.4f}")
        print(f"  Best Epoch: {t['best_epoch']}")
        print(f"  Parameters: {t['num_parameters']/1e6:.1f}M")
        print(f"  Avg Inference: {t['avg_inference_time_ms']:.1f} ms/batch")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()