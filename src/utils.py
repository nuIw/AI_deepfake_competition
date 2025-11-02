import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def create_dataloader(data_path, batch_size):
    """
    데이터셋과 데이터로더를 생성합니다. (CIFAR-10 예시)
    """
    # CIFAR-10의 평균과 표준편차
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # 학습 데이터 변환 (Augmentation 포함)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 검증/테스트 데이터 변환 (Augmentation 없음)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_model(model_name, num_classes=10):
    """
    모델을 생성합니다. (torchvision 모델 예시)
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif model_name == 'vgg11':
        model = models.vgg11(weights=None, num_classes=num_classes)
    # ... 다른 모델들 추가 ...
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
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