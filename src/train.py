import os
import sys
from pathlib import Path

# src를 Python path에 추가 (모든 import 전에 실행)
src_dir = Path(__file__).parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import random
import time
import wandb
from tqdm import tqdm
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn.metrics import f1_score

@hydra.main(config_path='../configs',config_name='config.yaml',version_base=None)
def main(cfg: DictConfig):
    # Hydra가 working directory를 바꾸므로 다시 한번 확인
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    print(f'Python path includes: {src_dir}')
    print(f'Current working directory: {os.getcwd()}')
    print('Training with config:')
    print(OmegaConf.to_yaml(cfg))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    run = wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, 
               config=OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True))
    
    accelerator = Accelerator()
    
    # model instantiate (name은 **kwargs로 자동 무시됨)
    model = instantiate(cfg.model)
    model_name = cfg.model.name  # name 접근
    
    # 체크포인트 디렉토리 생성: 모델명/실험명/run이름
    checkpoint_base = cfg.checkpoint.save_dir
    checkpoint_dir = os.path.join(
        checkpoint_base,
        model_name,           # freqnet
        cfg.exp_name,         # baseline
        wandb.run.name        
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {checkpoint_dir}')
    print(f'  Model: {model_name}')
    print(f'  Experiment: {cfg.exp_name}')
    print(f'  Run: {wandb.run.name}')
    
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    criterion = instantiate(cfg.criterion)
    
    train_loader = instantiate(cfg.data.train_loader)
    val_loader = instantiate(cfg.data.val_loader)
    
    wandb.watch(model, criterion, log='all')
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(cfg.run.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train(model, optimizer, criterion, train_loader, accelerator, epoch)
        val_loss, val_acc, val_f1 = val(model, criterion, val_loader, accelerator, epoch)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Learning rate 로깅
        current_lr = optimizer.param_groups[0]['lr']
        
        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch,
                'epoch_time': epoch_time,
                'learning_rate': current_lr
            }, step=epoch)
            
            print(f'\nEpoch {epoch}/{cfg.run.epochs} - {epoch_time:.2f}s - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f} - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f} - LR: {current_lr:.6f}')
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Best loss 기준 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}_best_loss.pth')
                
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f'✓ Best loss model saved at epoch {epoch} to {save_path}')
                
                # wandb에도 업로드
                wandb.save(save_path)
            
            # Best accuracy 기준 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}_best_acc.pth')
                
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f'✓ Best accuracy model saved at epoch {epoch} to {save_path}')
                
                # wandb에도 업로드
                wandb.save(save_path)
    
    # 최종 모델 저장
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, f'{model_name}_final.pth')
        torch.save(unwrapped_model.state_dict(), save_path)
        print(f'✓ Final model saved to {save_path}')
        
        # wandb에도 업로드
        wandb.save(save_path)
        
        wandb.finish()
    
    
def train(model, optimizer, criterion, train_loader, accelerator, epoch):
    model.train()
    cum_loss = 0.0
    total_samples = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    # Progress bar는 main process에서만 표시
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    else:
        pbar = train_loader
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.float()  # BCEWithLogitsLoss requires float targets
        loss = criterion(outputs.squeeze(), targets)
        cum_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Accuracy 계산 (Binary classification)
        predicted = (outputs.squeeze() > 0).float()
        correct += (predicted == targets).sum().item()
        
        # F1 score를 위한 예측/타겟 저장
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        accelerator.backward(loss)
        optimizer.step()
        
        # Progress bar 업데이트 (누적 평균 loss와 accuracy만 표시)
        if accelerator.is_main_process:
            pbar.set_postfix({
                'loss': f'{cum_loss / total_samples:.4f}',
                'acc': f'{100.0 * correct / total_samples:.2f}%'
            })
    
    total_loss = torch.tensor(cum_loss, device = accelerator.device)
    total_samples = torch.tensor(total_samples, device = accelerator.device)
    total_correct = torch.tensor(correct, device = accelerator.device)
    
    total_loss_tensor = accelerator.reduce(total_loss, reduction='sum')
    total_samples_tensor = accelerator.reduce(total_samples, reduction='sum')
    total_correct_tensor = accelerator.reduce(total_correct, reduction='sum')
    
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    accuracy = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()
    
    # 최종 F1 score 계산 (Multi-GPU 고려)
    all_predictions_tensor = torch.tensor(all_predictions, device=accelerator.device)
    all_targets_tensor = torch.tensor(all_targets, device=accelerator.device)
    
    all_predictions_gathered = accelerator.gather_for_metrics(all_predictions_tensor)
    all_targets_gathered = accelerator.gather_for_metrics(all_targets_tensor)
    
    final_f1 = f1_score(
        all_targets_gathered.cpu().numpy(), 
        all_predictions_gathered.cpu().numpy(), 
        average='macro',  # 두 클래스(0, 1) 평균
        zero_division=0
    )
    
    wandb.log({
        'train/loss': avg_loss,
        'train/accuracy': accuracy,
        'train/f1_score': final_f1}, step=epoch)
    return avg_loss, accuracy, final_f1
        
def val(model, criterion, val_loader, accelerator, epoch):
    model.eval()
    cum_loss = 0.0
    total_samples = 0
    correct = 0
    all_predictions = []
    all_targets = []
    
    # Progress bar는 main process에서만 표시
    if accelerator.is_main_process:
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            outputs = model(inputs)
            targets = targets.float()  # BCEWithLogitsLoss requires float targets
            loss = criterion(outputs.squeeze(), targets)
            cum_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Accuracy 계산 (Binary classification)
            predicted = (outputs.squeeze() > 0).float()
            correct += (predicted == targets).sum().item()
            
            # F1 score를 위한 예측/타겟 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Progress bar 업데이트 (누적 평균 loss와 accuracy만 표시)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    'loss': f'{cum_loss / total_samples:.4f}',
                    'acc': f'{100.0 * correct / total_samples:.2f}%'
                })
            
    total_loss = torch.tensor(cum_loss, device = accelerator.device)
    total_samples = torch.tensor(total_samples, device = accelerator.device)
    total_correct = torch.tensor(correct, device = accelerator.device)
    
    total_loss_tensor = accelerator.reduce(total_loss, reduction='sum')
    total_samples_tensor = accelerator.reduce(total_samples, reduction='sum')
    total_correct_tensor = accelerator.reduce(total_correct, reduction='sum')
    
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    accuracy = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()
    
    # 최종 F1 score 계산 (Multi-GPU 고려)
    all_predictions_tensor = torch.tensor(all_predictions, device=accelerator.device)
    all_targets_tensor = torch.tensor(all_targets, device=accelerator.device)
    
    all_predictions_gathered = accelerator.gather_for_metrics(all_predictions_tensor)
    all_targets_gathered = accelerator.gather_for_metrics(all_targets_tensor)
    
    final_f1 = f1_score(
        all_targets_gathered.cpu().numpy(), 
        all_predictions_gathered.cpu().numpy(), 
        average='macro',  # 두 클래스(0, 1) 평균
        zero_division=0
    )
    
    wandb.log({
        'val/loss': avg_loss,
        'val/accuracy': accuracy,
        'val/f1_score': final_f1}, step=epoch)
    return avg_loss, accuracy, final_f1


if __name__ == '__main__':
    main()