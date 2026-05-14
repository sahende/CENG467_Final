"""
CENG 467 - Knowledge Distillation for Task-Specific NLU
Configuration file
"""

import torch

class Config:
    # Model settings
    TEACHER_MODEL = "bert-base-uncased" 
    STUDENT_NUM_LAYERS = 6
    STUDENT_HIDDEN_SIZE = 768
    STUDENT_NUM_HEADS = 12
    
    # Training settings
    BATCH_SIZE = 16
    TEACHER_BATCH_SIZE = 8          
    GRADIENT_ACCUMULATION_STEPS = 2  
    LEARNING_RATE = 1e-5
    STUDENT_LR = 5e-5
    NUM_EPOCHS = 10
    MAX_SEQ_LENGTH = 128             
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10      
    EARLY_STOPPING_MIN_DELTA = 0 
    
    # Optimizer settings
    WEIGHT_DECAY = 0.01              
    WARMUP_RATIO = 0        
    
    # Distillation settings
    TEMPERATURE = 4.0
    ALPHA = 0.5
    
    # Data settings
    TASKS = ["rte", "mrpc"]
    VAL_SPLIT = 0.2
    
    # Paths
    MODEL_SAVE_PATH = "./models"
    RESULTS_PATH = "./results"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    SEED = 42
