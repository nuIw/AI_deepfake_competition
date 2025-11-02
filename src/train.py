import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.cuda as cuda
import numpy as np
import wandb
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from utils import (
    create_dataloader,
    create_model,
    create_optimizer,
    create_scheduler,
    create_criterion
)

@hydra.main(config_path='../configs',config_name='train.yaml',version_base=None)
def main(cfg: DictConfig):
    print('Training with config:')
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project = cfg.project_name,
        entity = cfg.entity,
        config = OmegaConf.to_container(cfg, resolve=True)
    )
    
    accelerator = Accelerator()
    
    train_loader, val_loader = create_dataloader(
        data_path = cfg.data.path,
        batch_size = cfg.data.batch_size
    )
    
    model = create_model(
        model_name = cfg.model.name
    )
    
    
    optimizer = create_optimizer(
        model = model,
        optimizer_name = cfg.optimizer.name,
        lr = cfg.optimizer.lr,
        weight_decay = cfg.optimizer.weight_decay
    )
    
    scheduler = create_scheduler(
        optimizer = optimizer,
        scheduler_name = cfg.scheduler.name,
        epochs = cfg.epochs
    )
    
    criterion = create_criterion(
        criterion_name = cfg.criterion.name
    )
    wandb.watch(model, criterion, log='all')
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        train_loss = train(model, optimizer, criterion, train_loader, accelerator, epoch, batch_size)
        val_loss = val(model, criterion, val_loader, accelerator, epoch, batch_size)
        scheduler.step()
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = wandb.run.dir
                save_path = os.path.join(save_dir, f'{cfg.model.name}_{epoch}_best.pth')
                
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f'Best model saved at epoch {epoch} to {save_path}')
    if accelerator.is_main_process:
        wandb.finish()
    
    
def train(model, optimizer, criterion, train_loader, accelerator, epoch, batch_size):
    model.train()
    cum_loss = 0.0
    total_samples = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        cum_loss += loss.item() * batch_size
        total_samples += batch_size
        accelerator.backward(loss)
        optimizer.step()
    total_loss = torch.tensor(cum_loss, device = accelerator.device)
    total_samples = torch.tensor(total_samples, device = accelerator.device)
    
    total_loss_tensor = accelerator.reduce(total_loss, reduction='sum')
    total_samples_tensor = accelerator.reduce(total_samples, reduction='sum')
    
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    wandb.log({
        'train/loss': avg_loss}, step=epoch)
    return avg_loss
        
def val(model, criterion, val_loader, accelerator, epoch, batch_size):
    model.eval()
    cum_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            cum_loss += loss.item() * batch_size
            total_samples += batch_size
    total_loss = torch.tensor(cum_loss, device = accelerator.device)
    total_samples = torch.tensor(total_samples, device = accelerator.device)
    
    total_loss_tensor = accelerator.reduce(total_loss, reduction='sum')
    total_samples_tensor = accelerator.reduce(total_samples, reduction='sum')
    
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    wandb.log({
        'val/loss': avg_loss}, step=epoch)
    return avg_loss