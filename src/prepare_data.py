"""
CENG 467 - Data Preparation for GLUE Tasks (RTE, MRPC)
Knowledge Distillation for Task-Specific NLU
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from config import Config

# Reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

class GLUEDataset(Dataset):
    """Custom Dataset class for GLUE sentence-pair tasks."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

def download_glue_data(task_name):
    """
    Downloads a GLUE task dataset using Hugging Face datasets library.
    
    Args:
        task_name (str): GLUE task name ('rte' or 'mrpc')
    
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    print(f"Downloading GLUE dataset: {task_name}...")
    
    try:
        dataset = load_dataset("glue", task_name)
        print(f"Successfully loaded {task_name} dataset!")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
        # Show sample
        print(f"\nSample from {task_name}:")
        print(f"Sentence 1: {dataset['train'][0]['sentence1']}")
        print(f"Sentence 2: {dataset['train'][0]['sentence2']}")
        print(f"Label: {dataset['train'][0]['label']}")
        
        return dataset
    except Exception as e:
        print(f"Error downloading {task_name}: {e}")
        raise

def tokenize_and_format(dataset, tokenizer, task_name):
    """
    Tokenizes GLUE sentence-pair data and returns PyTorch DataLoaders.
    
    Args:
        dataset: Hugging Face DatasetDict
        tokenizer: Hugging Face tokenizer
        task_name (str): GLUE task name
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    num_labels = 2  # RTE ve MRPC binary classification
    
    def preprocess_function(examples):
        """Tokenize sentence pairs."""
        return tokenizer(
            examples['sentence1'],
            examples['sentence2'],
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_SEQ_LENGTH
        )
    
    print(f"Tokenizing {task_name} dataset...")
    
    # Tokenize all splits
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Prepare for PyTorch tensors
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    
    # Create validation split from training data (20%)
    train_data = tokenized_datasets['train']
    train_size = int((1 - Config.VAL_SPLIT) * len(train_data))
    val_size = len(train_data) - train_size
    
    train_subset, val_subset = random_split(
        train_data, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Test set (GLUE validation set, as labels are not public for test set)
    test_loader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def prepare_all_tasks():
    """
    Downloads and prepares all tasks (RTE, MRPC) for the project.
    
    Returns:
        Dictionary containing DataLoaders for each task
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL)
    
    task_dataloaders = {}
    
    for task in Config.TASKS:
        print(f"\n{'='*50}")
        print(f"Preparing {task.upper()} task")
        print('='*50)
        
        # Download
        dataset = download_glue_data(task)
        
        # Tokenize and create loaders
        train_loader, val_loader, test_loader = tokenize_and_format(
            dataset, tokenizer, task
        )
        
        task_dataloaders[task] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    return task_dataloaders, tokenizer

if __name__ == "__main__":
    print("Starting GLUE data preparation...")
    print(f"Tasks: {Config.TASKS}")
    
    task_dataloaders, tokenizer = prepare_all_tasks()
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)