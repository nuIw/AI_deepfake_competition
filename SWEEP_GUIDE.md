# WandB Sweep 사용 가이드

## 개요

WandB Sweep을 사용하여 자동으로 최적의 하이퍼파라미터를 찾습니다.

---

## 사용 방법

### 1단계: Sweep Config 수정 (선택)

`configs/sweep.yaml` 파일에서 튜닝할 하이퍼파라미터를 설정합니다.

```yaml
parameters:
  optimizer.lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  
  data.batch_size:
    values: [16, 32, 64]
```

### 2단계: Sweep 초기화

```bash
python sweep_runner.py init
```

출력 예시:
```
Sweep ID: abc123xyz
Agent 실행 명령어:
  python sweep_runner.py agent abc123xyz
```

### 3단계: Agent 실행

```bash
# Sweep ID로 agent 실행
python sweep_runner.py agent abc123xyz

# 10번만 실행
python sweep_runner.py agent abc123xyz --count 10
```

---

## Colab에서 사용

```python
# 1. 초기화
!python sweep_runner.py init
# Sweep ID를 복사하세요

# 2. Agent 실행 (예: abc123xyz)
!python sweep_runner.py agent abc123xyz --count 5
```

---

## 병렬 실행

여러 agent를 동시에 실행하여 빠르게 탐색할 수 있습니다:

```bash
# Terminal 1
python sweep_runner.py agent abc123xyz

# Terminal 2
python sweep_runner.py agent abc123xyz

# Terminal 3
python sweep_runner.py agent abc123xyz
```

Colab에서는 여러 노트북을 열어서 동시 실행 가능합니다.

---

## Sweep Config 옵션

### Method (탐색 방법)

- `grid`: 모든 조합 탐색 (느림, 확실함)
- `random`: 랜덤 샘플링 (빠름, 대략적)
- `bayes`: Bayesian 최적화 (추천, 효율적)

### Parameter 분포

```yaml
# 고정값
param_name:
  value: 0.001

# 리스트에서 선택
param_name:
  values: [0.001, 0.01, 0.1]

# 균등 분포
param_name:
  distribution: uniform
  min: 0
  max: 1

# 로그 균등 분포 (learning rate에 좋음)
param_name:
  distribution: log_uniform_values
  min: 0.0001
  max: 0.1

# 정수 범위
param_name:
  distribution: int_uniform
  min: 16
  max: 128
```

### Metric

최적화할 metric 설정:

```yaml
metric:
  name: val/f1_score  # wandb에 로깅된 metric 이름
  goal: maximize      # maximize 또는 minimize
```

### Early Terminate

성능이 안 좋은 run을 조기 종료:

```yaml
early_terminate:
  type: hyperband
  min_iter: 10  # 최소 10 epoch는 실행
  eta: 2
  s: 2
```

---

## Tips

### 1. 적은 epoch로 빠르게 탐색

```yaml
# sweep.yaml
parameters:
  run.epochs:
    value: 20  # 빠른 탐색을 위해 epoch 줄이기
```

### 2. 가장 중요한 파라미터만 튜닝

```yaml
# 중요도 순서: lr > batch_size > weight_decay
parameters:
  optimizer.lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  
  # 나머지는 고정
  data.batch_size:
    value: 32
```

### 3. 결과 확인

WandB UI에서:
- Parallel Coordinates Plot: 파라미터 간 관계
- Importance: 어떤 파라미터가 중요한지
- Best Runs: 최고 성능 run들

---

## 문제 해결

### "No module named 'wandb'"
```bash
pip install wandb
```

### Sweep이 멈춤
- Ctrl+C로 중단 후 다시 실행
- 같은 sweep_id로 agent 재시작 가능

### 메모리 부족
```yaml
# batch_size 줄이기
parameters:
  data.batch_size:
    values: [8, 16]
```

---

## 예제

### Learning Rate 찾기
```yaml
method: bayes
metric:
  name: val/loss
  goal: minimize

parameters:
  optimizer.lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1
  
  # 나머지 고정
  data.batch_size:
    value: 32
  exp_name:
    value: "lr_sweep"
```

### 전체 튜닝
```yaml
method: bayes
metric:
  name: val/f1_score
  goal: maximize

parameters:
  optimizer.lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  
  optimizer.weight_decay:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
  
  data.batch_size:
    values: [16, 32, 64]
  
  scheduler.eta_min:
    distribution: log_uniform_values
    min: 1e-7
    max: 1e-4
```

---

## 참고 자료

- [WandB Sweep 문서](https://docs.wandb.ai/guides/sweeps)
- [Hydra 문서](https://hydra.cc/docs/intro/)
