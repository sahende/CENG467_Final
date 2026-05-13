"""
CENG 467 - Model Definitions
Knowledge Distillation for Task-Specific NLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    BertConfig
)
from config import Config

def get_teacher_model(task_name, num_labels=2):
    """
    Loads pre-trained BERT-base as the teacher model.
    
    Args:
        task_name (str): GLUE task name (for logging)
        num_labels (int): Number of output classes
    
    Returns:
        Hugging Face model for sequence classification
    """
    print(f"Loading teacher model: {Config.TEACHER_MODEL} for {task_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.TEACHER_MODEL,
        num_labels=num_labels,
        output_hidden_states=True,    # For feature distillation
        output_attentions=True         # For attention transfer
    )
    
    print(f"Teacher model parameters: {count_parameters(model):,}")
    return model

def get_student_model(num_labels=2):
    """
    Creates a smaller 6-layer student model based on BERT architecture.
    
    Args:
        num_labels (int): Number of output classes
    
    Returns:
        Smaller BERT-like model
    """
    print("Creating student model (6-layer BERT)...")
    
    # Use BERT config but with fewer layers
    config = BertConfig.from_pretrained(
        Config.TEACHER_MODEL,
        num_labels=num_labels,
        num_hidden_layers=Config.STUDENT_NUM_LAYERS,  # 6 instead of 12
        output_hidden_states=True,
        output_attentions=True
    )
    
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Initialize weights from scratch (not from pre-trained)
    model.init_weights()
    
    print(f"Student model parameters: {count_parameters(model):,}")
    return model

def count_parameters(model):
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    Loss = ALPHA * CE(student_logits, labels) + 
           (1 - ALPHA) * T^2 * KL(softmax(teacher/T), softmax(student/T))
    """
    
    def __init__(self, temperature=Config.TEMPERATURE, alpha=Config.ALPHA):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Output from student model
            teacher_logits: Output from teacher model (no_grad)
            labels: True labels
        
        Returns:
            Combined loss value
        """
        # Hard loss: cross-entropy with true labels
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between softmax distributions
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by T^2
        
        # Combine losses
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return total_loss, ce_loss, kl_loss