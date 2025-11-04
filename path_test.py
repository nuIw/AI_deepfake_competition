"""
체크포인트 경로 저장 테스트 스크립트
"""
import os
import sys
from pathlib import Path
import torch
from datetime import datetime

# src를 Python path에 추가
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='configs', config_name='config.yaml', version_base=None)
def test_checkpoint_path(cfg: DictConfig):
    print("="*60)
    print("체크포인트 경로 테스트")
    print("="*60)
    
    # Config 출력
    print("\n[Config]")
    print(f"  checkpoint.save_dir: {cfg.checkpoint.save_dir}")
    print(f"  exp_name: {cfg.exp_name}")
    print(f"  model.name: {cfg.model.name}")
    
    # 가짜 wandb run name 생성
    fake_run_name = f"test-run-{datetime.now().strftime('%H%M%S')}"
    print(f"\n[WandB Run Name (simulated)]")
    print(f"  {fake_run_name}")
    
    # 체크포인트 디렉토리 생성: 모델명/실험명/run이름
    checkpoint_base = cfg.checkpoint.save_dir
    checkpoint_dir = os.path.join(
        checkpoint_base,
        cfg.model.name,      # freqnet
        cfg.exp_name,        # baseline
        fake_run_name        # test-run-153042
    )
    
    print(f"\n[생성될 디렉토리 구조]")
    print(f"  {checkpoint_dir}")
    
    # 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\n✓ 디렉토리 생성 완료!")
    
    # 테스트 파일 저장
    test_files = [
        f'{cfg.model.name}_epoch5_best_loss.pth',
        f'{cfg.model.name}_epoch10_best_acc.pth',
        f'{cfg.model.name}_final.pth'
    ]
    
    print(f"\n[테스트 파일 저장]")
    for filename in test_files:
        filepath = os.path.join(checkpoint_dir, filename)
        
        # 더미 모델 state dict 생성
        dummy_state = {
            'model': 'test',
            'epoch': 0,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(dummy_state, filepath)
        print(f"  ✓ {filename}")
    
    # 전체 경로 트리 출력
    print(f"\n[디렉토리 트리]")
    print_tree(checkpoint_base, max_depth=4)
    
    print(f"\n{'='*60}")
    print("테스트 완료!")
    print(f"{'='*60}")
    print(f"\n저장 위치: {checkpoint_dir}")
    
    # 정리할지 물어보기
    print(f"\n테스트 파일을 삭제하시겠습니까? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(os.path.join(checkpoint_base, cfg.model.name, cfg.exp_name, fake_run_name))
            print("✓ 테스트 파일 삭제 완료")
    except:
        print("(입력 건너뜀)")


def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """디렉토리 트리 출력"""
    if current_depth >= max_depth:
        return
    
    try:
        directory = Path(directory)
        if not directory.exists():
            print(f"{prefix}(디렉토리 없음)")
            return
        
        entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, entry in enumerate(entries[:10]):  # 최대 10개만 표시
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            if entry.is_dir():
                print(f"{prefix}{current_prefix}{entry.name}/")
                print_tree(entry, prefix + next_prefix, max_depth, current_depth + 1)
            else:
                size = entry.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                print(f"{prefix}{current_prefix}{entry.name} ({size_str})")
        
        if len(entries) > 10:
            print(f"{prefix}... ({len(entries) - 10} more items)")
    
    except Exception as e:
        print(f"{prefix}(접근 불가: {e})")


if __name__ == '__main__':
    test_checkpoint_path()

