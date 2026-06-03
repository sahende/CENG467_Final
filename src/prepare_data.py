"""
CENG 467 - Data Preparation for GLUE Tasks
Knowledge Distillation with Domain Adaptation

FIX: Bypass HuggingFace cache for subsampling by reloading fresh each time
"""

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from config import Config

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def prepare_single_task(task_name, tokenizer, max_train_samples=None):
    """
    Prepare DataLoaders for a single GLUE task.
    
    Args:
        max_train_samples: If set, randomly subsample training data to this size
    """
    print(f"  Loading {task_name.upper()}...")
    
    # ================================================================
    # FIX: Fresh load every time (no cache interference)
    # ================================================================
    import datasets
    datasets.disable_caching()
    
    dataset = load_dataset("glue", task_name)
    datasets.enable_caching()
    
    original_size = len(dataset['train'])
    print(f"    Original train size: {original_size}")
    
    # ================================================================
    # CONTROLLED SUBSAMPLING
    # ================================================================
    if max_train_samples is not None and original_size > max_train_samples:
        rng = np.random.RandomState(Config.SEED)
        indices = rng.choice(original_size, max_train_samples, replace=False)
        dataset['train'] = dataset['train'].select(indices.tolist())
        print(f"    ⚠ Subsampled: {original_size} → {max_train_samples}")
    else:
        print(f"    Using full train: {original_size} samples")
    
    is_single_sentence = task_name in ["sst2", "cola"]
    
    def preprocess_function(examples):
        if is_single_sentence:
            return tokenizer(
                examples['sentence'],
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_SEQ_LENGTH
            )
        else:
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                truncation=True,
                padding='max_length',
                max_length=Config.MAX_SEQ_LENGTH
            )
    
    tokenized = dataset.map(preprocess_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    
    # Create train/val split
    train_data = tokenized['train']
    train_size = int((1 - Config.VAL_SPLIT) * len(train_data))
    val_size = len(train_data) - train_size
    
    train_subset, val_subset = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    return {
        'train': DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False),
        'test': DataLoader(
            tokenized['validation'],
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        ),
        'train_dataset': train_subset,
        'val_dataset': val_subset,
        'train_size': len(train_subset)
    }


def prepare_all_tasks(task_list=None, task_subsample_sizes=None):
    """
    Prepare data for given task list.
    
    Args:
        task_subsample_sizes: Dict mapping task_name → max_train_samples
                             e.g., {"cola": 3668, "sst2": 5000}
    """
    if task_list is None:
        task_list = Config.TASKS
    
    if task_subsample_sizes is None:
        task_subsample_sizes = {}
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL)
    print(f"Preparing tasks: {task_list}")
    if task_subsample_sizes:
        print(f"  Subsampling config: {task_subsample_sizes}")
    
    task_dataloaders = {}
    for task in task_list:
        max_samples = task_subsample_sizes.get(task, None)
        task_dataloaders[task] = prepare_single_task(task, tokenizer, max_train_samples=max_samples)
    
    return task_dataloaders, tokenizer


def prepare_combined_tasks(task_list, task_subsample_sizes=None):
    """Combine multiple tasks into one DataLoader."""
    if task_subsample_sizes is None:
        task_subsample_sizes = {}
    
    tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL)
    print(f"Preparing COMBINED tasks: {task_list}")
    
    task_data = {}
    for task in task_list:
        max_samples = task_subsample_sizes.get(task, None)
        task_data[task] = prepare_single_task(task, tokenizer, max_train_samples=max_samples)
    
    train_datasets = [task_data[t]['train_dataset'] for t in task_list]
    combined_train = ConcatDataset(train_datasets)
    
    val_datasets = [task_data[t]['val_dataset'] for t in task_list]
    combined_val = ConcatDataset(val_datasets)
    
    combined_loaders = {
        'train': DataLoader(combined_train, batch_size=Config.BATCH_SIZE, shuffle=True),
        'val': DataLoader(combined_val, batch_size=Config.BATCH_SIZE, shuffle=False),
    }
    
    per_task_test = {t: task_data[t]['test'] for t in task_list}
    
    print(f"  Combined train: {len(combined_train)}, val: {len(combined_val)}")
    
    return combined_loaders, per_task_test, tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("  GLUE Data Preparation (FRESH LOAD)")
    print("=" * 60)
    
    task_dataloaders, _ = prepare_all_tasks(
        task_list=["rte", "mrpc", "cola", "sst2"],
        task_subsample_sizes={
            "cola": 3668,
            "sst2": 5000,
        }
    )
    print("\n✓ Data preparation complete!")