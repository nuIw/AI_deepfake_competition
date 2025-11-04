## 기능

1. **새 아티팩트 생성**: 디렉토리를 아티팩트로 저장
2. **아티팩트 확장**: 기존 아티팩트에 새 데이터를 추가하여 새 버전 생성
3. **아티팩트 다운로드**: 저장된 아티팩트를 로컬로 다운로드

## 사용법

### CLI 사용 (Colab에서)

#### 새 아티팩트 생성

```bash
%cd /content/MIP-team11
!python src/raw.py \
    --project MIP-0 \
    --entity dmachine-kyung-hee-university \
    create \
    --name dataset-v1 \
    --type dataset \
    --dir /cotnet/data \
    --description "Initial dataset"
```

#### 기존 아티팩트 확장

```bash
%cd /content/MIP-team11
!python src/raw.py \
    --project MIP-0 \
    --entity dmachine-kyung-hee-university \
    extend \
    --base dataset-v1:latest \
    --name dataset-v2 \
    --dir /cotnet/data/additional \
    --description "Extended dataset"
```

#### 아티팩트 다운로드

```bash
%cd /content/MIP-team11
python src/raw.py \
    --project MIP-0 \
    --entity dmachine-kyung-hee-university \
    download \
    --name dataset-v2:latest \
    --path /content/data
```

## 주요 매개변수

### ArtifactManager 초기화

- `project_name`: wandb 프로젝트 이름 (필수)
- `entity`: wandb entity (팀 또는 사용자 이름, 선택)

### create_artifact

- `artifact_name`: 아티팩트 이름 (필수)
- `artifact_type`: 아티팩트 타입 (필수, 예: "dataset", "model", "result")
- `directory_path`: 저장할 디렉토리 경로 (필수)
- `description`: 아티팩트 설명 (선택)
- `metadata`: 메타데이터 dict (선택)

### extend_artifact

- `base_artifact_name`: 기존 아티팩트 이름 (필수, 예: "my-dataset:v0" 또는 "my-dataset:latest")
- `new_artifact_name`: 새 아티팩트 이름 (필수)
- `directory_path`: 추가할 디렉토리 경로 (필수)
- `artifact_type`: 아티팩트 타입 (선택, None이면 기존과 동일)
- `description`: 설명 (선택)
- `metadata`: 메타데이터 dict (선택)

### download_artifact

- `artifact_name`: 다운로드할 아티팩트 이름 (필수, 예: "my-dataset:latest")
- `download_path`: 저장할 경로 (필수)

## 아티팩트 버전 관리

wandb는 자동으로 아티팩트의 버전을 관리합니다:

- 같은 이름으로 새 아티팩트를 생성하면 v0, v1, v2... 순으로 버전이 증가
- `:latest` 별칭을 사용하면 항상 최신 버전을 참조
- `:v0`, `:v1` 등으로 특정 버전을 참조 가능

예제:
```python
# 최신 버전 사용
artifact = manager.download_artifact("dataset:latest", "./data")

# 특정 버전 사용
artifact = manager.download_artifact("dataset:v0", "./data")
```

## 실전 워크플로우 예제

### 데이터셋 점진적 구축

```python
from src.raw import ArtifactManager

# 1. 초기 데이터셋 생성
manager = ArtifactManager(project_name="deepfake-detection")
manager.create_artifact(
    artifact_name="training-data",
    artifact_type="dataset",
    directory_path="./data/batch_001",
    metadata={"batch": 1, "samples": 1000}
)
manager.finish()

# 2. 새로운 배치 추가 (버전 v1)
manager = ArtifactManager(project_name="deepfake-detection")
manager.extend_artifact(
    base_artifact_name="training-data:latest",
    new_artifact_name="training-data",  # 같은 이름 = 새 버전
    directory_path="./data/batch_002",
    metadata={"batch": 2, "total_samples": 2000}
)
manager.finish()

# 3. 학습 시 최신 데이터 사용
manager = ArtifactManager(project_name="deepfake-detection")
data_path = manager.download_artifact(
    artifact_name="training-data:latest",
    download_path="./data/current"
)
manager.finish()

# 이제 data_path로 학습 진행
```

## Tips

1. **메타데이터 활용**: 각 아티팩트에 샘플 수, 날짜, 출처 등의 메타데이터를 저장하면 추적이 용이합니다.

2. **버전 전략**: 
   - 증분 업데이트: 같은 이름으로 계속 새 버전 생성
   - 독립 버전: 각 배치마다 고유한 이름 사용

3. **타입 분류**: 
   - `dataset`: 원시 데이터
   - `processed-data`: 전처리된 데이터
   - `model`: 학습된 모델
   - `result`: 실험 결과

4. **CI/CD 통합**: CLI를 사용하면 자동화된 파이프라인에 쉽게 통합 가능

## 문제 해결

### wandb 로그인 오류
```bash
wandb login
```

### 대용량 파일 업로드
wandb는 기본적으로 대용량 파일을 지원하지만, 매우 큰 데이터셋의 경우 청크 단위로 나누는 것을 권장합니다.

### 권한 오류
entity와 project_name이 올바른지, 그리고 해당 프로젝트에 대한 쓰기 권한이 있는지 확인하세요.

## 참고 자료

- [wandb Artifacts 공식 문서](https://docs.wandb.ai/guides/artifacts)
- [wandb Python API](https://docs.wandb.ai/ref/python)

