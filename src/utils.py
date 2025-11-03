import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from model.py import *

def create_model(model_name, num_classes=10):
    """
    모델을 생성합니다. (torchvision 모델 예시)
    """
    if model_name.lower() == 'resnet18':
    
    return model

def create_optimizer(model, optimizer_name, lr, weight_decay=1e-4):
    """
    옵티마이저를 생성합니다.
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # ... 다른 옵티마이저들 추가 ...
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
        
    return optimizer

def create_scheduler(optimizer, scheduler_name, epochs):
    """
    학습률 스케줄러를 생성합니다.
    """
    if scheduler_name.lower() == 'cosine':
        # T_max를 epochs 수로 설정하는 것이 일반적입니다.
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name.lower() == 'step':
        # 30 에폭마다 0.1씩 감소
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # ... 다른 스케줄러들 추가 ...
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
        
    return scheduler

def create_criterion(criterion_name):
    """
    손실 함수를 생성합니다.
    """
    if criterion_name.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name.lower() == 'mse':
        criterion = nn.MSELoss()
    # ... 다른 손실 함수들 추가 ...
    else:
        raise ValueError(f"Unknown criterion name: {criterion_name}")
        
    return criterion