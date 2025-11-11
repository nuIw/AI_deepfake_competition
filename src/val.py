import os
import sys
from pathlib import Path
import argparse
import shutil

# src를 Python path에 추가
src_dir = Path(__file__).parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch.nn.functional as F
import wandb


def validate_with_threshold(model, val_loader, threshold=0.5, device='cuda'):
    """
    주어진 threshold로 validation set에 대해 평가
    
    Args:
        model: 평가할 모델
        val_loader: validation DataLoader
        threshold: prediction threshold (default: 0.5)
        device: 디바이스 (cuda/cpu)
    
    Returns:
        dict: 평가 결과 (accuracy, f1_macro, f1_weighted, per_class_f1, confusion matrix 등)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    print(f'\nEvaluating with threshold: {threshold}')
    print('='*60)
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Sigmoid를 적용하여 확률값 계산
            probabilities = torch.sigmoid(outputs.squeeze())
            
            # Threshold를 적용하여 예측값 계산
            predicted = (probabilities >= threshold).float()
            
            # 결과 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # numpy array로 변환
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    # Accuracy 계산
    accuracy = 100.0 * (all_predictions == all_targets).sum() / len(all_targets)
    
    # F1 scores 계산
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # 결과 출력
    print('\n' + '='*60)
    print(f'Validation Results (Threshold: {threshold})')
    print('='*60)
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'F1-Score (Macro): {f1_macro:.4f}')
    print(f'F1-Score (Weighted): {f1_weighted:.4f}')
    print(f'F1-Score (Class 0 - Real): {per_class_f1[0]:.4f}')
    print(f'F1-Score (Class 1 - Fake): {per_class_f1[1]:.4f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(classification_report(all_targets, all_predictions, 
                                target_names=['Real', 'Fake'], 
                                digits=4))
    print('='*60)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_class_0_real': per_class_f1[0],
        'f1_class_1_fake': per_class_f1[1],
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }


def scan_thresholds(model, val_loader, thresholds=None, device='cuda'):
    """
    여러 threshold에 대해 평가하여 최적의 threshold를 찾음
    
    Args:
        model: 평가할 모델
        val_loader: validation DataLoader
        thresholds: threshold 리스트 (None이면 0.1~0.9 사이 0.1 간격)
        device: 디바이스 (cuda/cpu)
    
    Returns:
        list: 각 threshold별 결과 리스트
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    
    # 먼저 모든 확률값을 얻음
    model.eval()
    all_probabilities = []
    all_targets = []
    
    print('Collecting predictions...')
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Forward pass'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs.squeeze())
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    print('\nScanning thresholds...')
    print('='*80)
    print(f'{"Threshold":<12} {"Accuracy":<12} {"F1-Macro":<12} {"F1-Class0":<12} {"F1-Class1":<12}')
    print('='*80)
    
    for threshold in thresholds:
        # Threshold 적용
        predictions = (all_probabilities >= threshold).astype(float)
        
        # Metrics 계산
        accuracy = 100.0 * (predictions == all_targets).sum() / len(all_targets)
        f1_macro = f1_score(all_targets, predictions, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_targets, predictions, average=None, zero_division=0)
        
        print(f'{threshold:<12.2f} {accuracy:<12.2f} {f1_macro:<12.4f} {per_class_f1[0]:<12.4f} {per_class_f1[1]:<12.4f}')
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_class_0': per_class_f1[0],
            'f1_class_1': per_class_f1[1]
        })
    
    print('='*80)
    
    # 최고 F1-Macro 찾기
    best_result = max(results, key=lambda x: x['f1_macro'])
    print(f'\nBest threshold: {best_result["threshold"]:.2f} (F1-Macro: {best_result["f1_macro"]:.4f})')
    
    return results


@hydra.main(config_path='../configs', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    """
    Main validation function
    
    CLI Usage:
        # 기본 threshold (0.5) 사용
        python src/val.py checkpoint_path=/path/to/checkpoint.pth
        
        # 특정 threshold 사용
        python src/val.py checkpoint_path=/path/to/checkpoint.pth threshold=0.7
        
        # 여러 threshold 스캔
        python src/val.py checkpoint_path=/path/to/checkpoint.pth scan_thresholds=true
        
        # 데이터 경로 변경
        python src/val.py checkpoint_path=/path/to/checkpoint.pth data.path=/path/to/data
        
        # Artifact 다운로드 (val만)
        python src/val.py checkpoint_path=/path/to/checkpoint.pth data.download_artifact=true
    """
    
    # Hydra가 working directory를 바꾸므로 다시 한번 확인
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    print('='*60)
    print('Validation Script')
    print('='*60)
    print(f'Python path includes: {src_dir}')
    print(f'Current working directory: {os.getcwd()}')
    
    # Config에서 파라미터 읽기
    checkpoint_path = cfg.get('checkpoint_path', None)
    threshold = cfg.get('threshold', 0.5)
    scan_thresholds_flag = cfg.get('scan_thresholds', False)
    
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided. Example: python src/val.py checkpoint_path=/path/to/model.pth")
    
    # WandB 초기화 (artifact 다운로드를 위해 필요)
    if cfg.data.get('download_artifact', False):
        run = wandb.init(
            project=cfg.wandb.project_name, 
            entity=cfg.wandb.entity,
            job_type='validation',
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        
        # Artifact 사용 로깅 (download_artifact 설정과 무관)
        if hasattr(cfg.data, 'raw_artifact') and cfg.data.raw_artifact:
            # raw_artifact가 리스트인지 단일 값인지 확인
            raw_artifacts = cfg.data.raw_artifact
            if isinstance(raw_artifacts, str):
                raw_artifacts = [raw_artifacts]
            
            # WandB에 artifact 사용 로깅
            for artifact_name in raw_artifacts:
                if ':' not in artifact_name:
                    artifact_name = f'{artifact_name}:latest'
                print(f'Logging artifact usage: {artifact_name}')
                run.use_artifact(artifact_name)
        
        # Artifact 다운로드 (val 폴더만)
        print('\n' + '='*60)
        print('Downloading validation data from artifacts...')
        print('='*60)
        
        raw_artifacts = cfg.data.raw_artifact
        if isinstance(raw_artifacts, str):
            raw_artifacts = [raw_artifacts]
        
        # 임시 다운로드 경로
        temp_download_dir = os.path.join(os.getcwd(), 'temp_datasets')
        # 최종 validation 데이터 경로
        final_val_dir = os.path.join(os.getcwd(), 'datasets', 'val')
        
        os.makedirs(final_val_dir, exist_ok=True)
        
        for i, artifact_name in enumerate(raw_artifacts):
            if ':' not in artifact_name:
                artifact_name = f'{artifact_name}:latest'
            
            print(f'\n  [{i+1}/{len(raw_artifacts)}] Processing artifact: {artifact_name}')
            artifact = run.use_artifact(artifact_name)
            
            # 임시 위치에 다운로드
            artifact_temp_dir = os.path.join(temp_download_dir, f'artifact_{i}')
            artifact.download(root=artifact_temp_dir)
            
            # val 폴더 찾기
            val_source = os.path.join(artifact_temp_dir, 'val')
            if not os.path.exists(val_source):
                print(f'    ⚠ Warning: No val directory found in {artifact_name}')
                continue
            
            # val 폴더의 클래스 디렉토리들을 최종 위치로 복사/병합
            for class_dir in os.listdir(val_source):
                class_source = os.path.join(val_source, class_dir)
                class_dest = os.path.join(final_val_dir, class_dir)
                
                if os.path.isdir(class_source):
                    os.makedirs(class_dest, exist_ok=True)
                    
                    # 해당 클래스의 모든 이미지 파일 복사
                    for img_file in os.listdir(class_source):
                        src_file = os.path.join(class_source, img_file)
                        # 파일명 중복 방지: artifact 인덱스 추가
                        dest_file = os.path.join(class_dest, f'artifact{i}_{img_file}')
                        
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dest_file)
                    
                    num_images = len([f for f in os.listdir(class_dest) if os.path.isfile(os.path.join(class_dest, f))])
                    print(f'    ✓ Copied class "{class_dir}": {num_images} images total')
        
        # 임시 다운로드 디렉토리 삭제
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)
            print(f'\n  ✓ Cleaned up temporary files')
        
        # cfg.data.path를 최종 데이터셋 경로로 업데이트
        OmegaConf.set_struct(cfg, False)
        cfg.data.path = os.path.join(os.getcwd(), 'datasets')
        OmegaConf.set_struct(cfg, True)
        
        print(f'\n  ✓ Total artifacts processed: {len(raw_artifacts)}')
        print(f'  ✓ Validation data path: {final_val_dir}')
        
        # 최종 통계
        total_images = 0
        for class_dir in os.listdir(final_val_dir):
            class_path = os.path.join(final_val_dir, class_dir)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
                total_images += num_images
                print(f'    - Class "{class_dir}": {num_images} images')
        print(f'    - Total: {total_images} images')
        print('='*60)
    else:
        print(f'\nSkipping artifact download. Using local path: {cfg.data.path}')
    
    # 데이터셋 구조 정리 (설정에서 활성화된 경우)
    if cfg.data.get('organize_labels', True):
        print('\n' + '='*50)
        print('Organizing dataset structure...')
        print('='*50)
        try:
            from organize_labels import organize_labels
            organize_labels(cfg.data.path, dry_run=False)
            print('✓ Dataset organization completed')
        except Exception as e:
            print(f'⚠ Warning: Dataset organization failed: {e}')
            print('  Continuing with original folder structure...')
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    # 모델 로드
    print(f'\nLoading model from: {checkpoint_path}')
    model = instantiate(cfg.model)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print('✓ Model loaded successfully')
    
    # Validation loader 생성
    print('\nCreating validation loader...')
    val_loader = instantiate(cfg.data.val_loader)
    print(f'✓ Validation set size: {len(val_loader.dataset)} images')
    
    # Threshold 스캔 모드
    if scan_thresholds_flag:
        print('\nScanning multiple thresholds...')
        results = scan_thresholds(model, val_loader, device=device)
    else:
        # 단일 threshold로 평가
        results = validate_with_threshold(model, val_loader, threshold=threshold, device=device)
    
    print('\n✓ Validation completed!')
    
    # WandB 종료
    if cfg.data.get('download_artifact', False):
        wandb.finish()


if __name__ == '__main__':
    main()

