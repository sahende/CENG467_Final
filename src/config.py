"""
Hierarchical Knowledge Distillation for Task-Specific NLU
"""

import os

from numpy import size
import torch

class Config:
    # Model settings
    TEACHER_MODEL = "bert-base-uncased" 
    STUDENT_NUM_LAYERS = 6
    STUDENT_HIDDEN_SIZE = 768
    STUDENT_NUM_HEADS = 12
    
    # Training settings
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2        # Effective batch 32
    LEARNING_RATE = 2e-5                   # Teacher
    STUDENT_LR = 5e-5                      # Student
    NUM_EPOCHS = 10
    MAX_SEQ_LENGTH = 128
    
    # Domain Adaptation settings
    ADAPT_EPOCHS = 10
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10           
    EARLY_STOPPING_MIN_DELTA = 0.0         
    
    # Optimizer settings
    WEIGHT_DECAY = 0.01              
    WARMUP_RATIO = 0.0                     
    
    # Distillation settings
    TEMPERATURE = 4.0
    ALPHA = 0.5
    
    # Data settings
    TASKS = ["cola" ]
    TARGET_TASKS = [ "cola"]
    VAL_SPLIT = 0.2
    
    # Data subsampling settings
    # Specify the dataset sizes to subsample for each task. If empty/not specified, uses full dataset.
    DATASET_SIZES = {"cola":2490}
    
    
    # Paths
    MODEL_SAVE_PATH = "./models"
    RESULTS_PATH = "./results"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42

    @classmethod
    def get_model_path(cls, task, category, filename, seed=None, dataset_size=None):
        """
        Builds the standard path:
        models/{base_model}/{task}/{dataset_size}/seed_{seed}/{category}/{filename}
        """
        base_model = cls.TEACHER_MODEL.split("-")[0]
        if dataset_size is None:
            dataset_size = str(cls.DATASET_SIZES.get(task, "full"))
        else:
            dataset_size = str(dataset_size)
        seed_str = str(seed) if seed is not None else str(cls.SEED)
        
        path = os.path.join(
            cls.MODEL_SAVE_PATH, 
            base_model,
            task,
            dataset_size,
            f"seed_{seed_str}",
            category
        )
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)
    
    @classmethod
    def get_task_sizes(cls, task):
        """
        Returns the list of dataset sizes for a given task.
        If no sizes are specified in DATASET_SIZES, returns a list with a single 'full' string.
        """
        if task in cls.DATASET_SIZES:
            return cls.DATASET_SIZES[task] if isinstance(cls.DATASET_SIZES[task], list) else [cls.DATASET_SIZES[task]]
        return ["full"]
    
    @classmethod
    def size_to_subsample(cls, task, size):
        """
        Converts a size value to a task_subsample_sizes dict for prepare_data.
        'full' -> {} (no subsampling)
        """
        if size == "full":
            return {}
        return {task: size if isinstance(size, int) else int(size)}