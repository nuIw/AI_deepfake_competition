"""
WandB Sweep Runner
하이퍼파라미터 자동 튜닝

사용법:
1. Sweep 초기화:
   python sweep_runner.py init

2. Agent 실행 (여러 터미널에서 동시 실행 가능):
   python sweep_runner.py agent <sweep_id>
"""
import sys
import os
import wandb
import yaml
from pathlib import Path

def load_sweep_config(config_path='configs/sweep.yaml'):
    """Sweep config 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def init_sweep():
    """Sweep 초기화 및 ID 반환"""
    print("=" * 60)
    print("WandB Sweep 초기화")
    print("=" * 60)
    
    # Sweep config 로드
    config = load_sweep_config()
    
    print("\n[Sweep Config]")
    print(f"  Method: {config['method']}")
    print(f"  Metric: {config['metric']['name']} ({config['metric']['goal']})")
    print(f"  Parameters to tune:")
    for param, settings in config['parameters'].items():
        if 'values' in settings:
            print(f"    - {param}: {settings['values']}")
        elif 'min' in settings and 'max' in settings:
            print(f"    - {param}: {settings['min']} ~ {settings['max']}")
        elif 'value' in settings:
            print(f"    - {param}: {settings['value']} (fixed)")
    
    # WandB login 확인
    try:
        wandb.login()
    except:
        print("\n⚠️  WandB 로그인이 필요합니다!")
        return None
    
    # Sweep 초기화
    sweep_id = wandb.sweep(
        sweep=config,
        project="MIP-0",  # config.yaml의 project_name과 동일하게
        entity='dmachine-kyung-hee-university'  # config.yaml의 entity와 동일하게
    )
    
    print("\n" + "=" * 60)
    print("✓ Sweep 초기화 완료!")
    print("=" * 60)
    print(f"\nSweep ID: {sweep_id}")
    print(f"\nAgent 실행 명령어:")
    print(f"  python sweep_runner.py agent {sweep_id}")
    print(f"\nWandB UI:")
    print(f"  https://wandb.ai/dmachine-kyung-hee-university/MIP-0/sweeps/{sweep_id}")
    print("\n" + "=" * 60)
    
    return sweep_id

def run_agent(sweep_id, count=None):
    """Sweep agent 실행"""
    print("=" * 60)
    print(f"WandB Sweep Agent 실행")
    print("=" * 60)
    print(f"\nSweep ID: {sweep_id}")
    if count:
        print(f"실행 횟수: {count}")
    else:
        print(f"실행 횟수: 무제한 (Ctrl+C로 중단)")
    print("\n" + "=" * 60 + "\n")
    
    # Train 함수를 sweep에서 실행
    def train_wrapper():
        """Sweep에서 호출할 train 함수"""
        import subprocess
        import sys
        
        # wandb.config에서 하이퍼파라미터 가져오기
        config = wandb.config
        
        # Hydra override 형식으로 변환
        overrides = []
        for key, value in config.items():
            overrides.append(f"{key}={value}")
        
        # train.py 실행
        cmd = [sys.executable, "src/train.py"] + overrides
        print(f"\n실행 명령어: {' '.join(cmd)}\n")
        
        subprocess.run(cmd, check=True)
    
    # Agent 실행
    wandb.agent(
        sweep_id,
        function=train_wrapper,
        count=count,
        project="MIP-0",
        entity='dmachine-kyung-hee-university'
    )

def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  1. Sweep 초기화: python sweep_runner.py init")
        print("  2. Agent 실행:   python sweep_runner.py agent <sweep_id>")
        print("  3. Agent 실행 (N번): python sweep_runner.py agent <sweep_id> --count N")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "init":
        init_sweep()
    
    elif command == "agent":
        if len(sys.argv) < 3:
            print("Sweep ID가 필요합니다!")
            print("사용법: python sweep_runner.py agent <sweep_id>")
            sys.exit(1)
        
        sweep_id = sys.argv[2]
        count = None
        
        # --count 옵션 파싱
        if len(sys.argv) > 3 and sys.argv[3] == "--count":
            count = int(sys.argv[4])
        
        run_agent(sweep_id, count)
    
    else:
        print(f"알 수 없는 명령어: {command}")
        print("사용 가능한 명령어: init, agent")
        sys.exit(1)

if __name__ == '__main__':
    main()
