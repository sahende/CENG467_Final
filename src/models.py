"""
CENG 467 - Model Definitions
Knowledge Distillation for Task-Specific NLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig
)
from config import Config


def get_teacher_model(task_name, num_labels=2):
    """Load pre-trained BERT-base as the teacher model."""
    print(f"Loading teacher model: {Config.TEACHER_MODEL} for {task_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.TEACHER_MODEL,
        num_labels=num_labels,
        output_hidden_states=False,
        output_attentions=False
    )
    
    print(f"  Teacher parameters: {count_parameters(model):,}")
    return model


def get_student_model(num_labels=2):
    """Create a 6-layer BERT student model with random initialization."""
    print("Creating student model (6-layer BERT, random init)...")
    
    config = BertConfig.from_pretrained(
        Config.TEACHER_MODEL,
        num_labels=num_labels,
        num_hidden_layers=Config.STUDENT_NUM_LAYERS,
        output_hidden_states=False,
        output_attentions=False
    )
    
    model = AutoModelForSequenceClassification.from_config(config)
    model.init_weights()
    
    print(f"  Student parameters: {count_parameters(model):,}")
    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DistillationLoss(nn.Module):
    """
    Logit-based Knowledge Distillation Loss (Hinton et al., 2015).
    
    L = α * CE(student_logits, ground_truth) 
        + (1-α) * T² * KL(softmax(teacher/T) || softmax(student/T))
    """
    
    def __init__(self, temperature=Config.TEMPERATURE, alpha=Config.ALPHA):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        """Compute combined distillation loss."""
        # Hard loss: cross-entropy with true labels
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between temperature-scaled distributions
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return total_loss, ce_loss, kl_loss