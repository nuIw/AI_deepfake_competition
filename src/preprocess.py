"""
데이터 전처리 스크립트

wandb Artifact를 사용하여 raw 데이터셋을 로드하고 전처리한 후
새로운 Artifact로 저장합니다.
"""

import os
import sys
from pathlib import Path

# src를 Python path에 추가
src_dir = Path(__file__).parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


@hydra.main(config_path='../configs', config_name='preprocess.yaml', version_base=None)
def main(cfg: DictConfig):
    """전처리 메인 함수"""
    
    print('전처리 시작')
    print('Configuration:')
    print(OmegaConf.to_yaml(cfg))
    
    # wandb 초기화 (job_type을 preprocess로 설정)
    with wandb.init(
        project=cfg.wandb.project_name, 
        entity=cfg.wandb.entity,
        job_type="preprocess",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ) as run:
        
        # 1. Raw Dataset Artifact 로드
        print(f"\n1. Raw Dataset Artifact 로드: {cfg.artifact.raw_artifact_name}")
        raw_artifact = run.use_artifact(
            cfg.artifact.raw_artifact_name, 
            type=cfg.artifact.raw_artifact_type
        )
        raw_path = raw_artifact.download()
        print(f"✓ Raw 데이터셋 다운로드 완료: {raw_path}")
        
        # 2. 전처리 파이프라인 정의 (Hydra instantiate 사용)
        print(f"\n2. 전처리 파이프라인 정의")
        preprocessing_transform = instantiate(cfg.preprocess.transform)
        print(f"✓ Transform 파이프라인:")
        for i, t in enumerate(preprocessing_transform.transforms):
            print(f"    {i+1}. {t.__class__.__name__}")
        
        # 3. 출력 디렉토리 설정
        print(f"\n3. 출력 디렉토리 설정")
        processed_dir = cfg.preprocess.output_dir
        os.makedirs(processed_dir, exist_ok=True)
        print(f"✓ 출력 디렉토리: {processed_dir}")
        
        # 4. train, val, test 각각 전처리
        print(f"\n4. 데이터셋 전처리 시작")
        
        # 전처리할 split 목록 (설정 파일에서 가져오거나 기본값 사용)
        splits = cfg.preprocess.get('splits', ['train', 'val', 'test'])
        
        total_images = 0
        all_classes = set()
        
        for split in splits:
            split_path = os.path.join(raw_path, split)
            
            # split 디렉토리가 존재하지 않으면 스킵
            if not os.path.exists(split_path):
                print(f"\n[{split}] 디렉토리가 없어 스킵: {split_path}")
                continue
            
            print(f"\n{'='*50}")
            print(f"[{split.upper()}] 전처리 시작")
            print(f"{'='*50}")
            
            # ImageFolder로 로드
            dataset = ImageFolder(root=split_path, transform=None)
            print(f"✓ {len(dataset)}개의 이미지 로드")
            print(f"✓ 클래스: {dataset.classes}")
            all_classes.update(dataset.classes)
            
            # 출력 디렉토리 생성 (processed_dir/split/class/)
            split_output_dir = os.path.join(processed_dir, split)
            for class_name in dataset.classes:
                class_dir = os.path.join(split_output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
            
            # 전처리 및 저장
            for i in tqdm(range(len(dataset)), desc=f"{split} 전처리"):
                # 원본 이미지와 레이블 로드
                img_pil, label = dataset[i]
                
                # 전처리 적용 (PIL -> Tensor -> PIL)
                processed_img_pil = preprocessing_transform(img_pil)
                
                # 저장 경로 생성
                class_name = dataset.classes[label]
                target_dir = os.path.join(split_output_dir, class_name)
                
                # 원본 파일명 사용
                original_filename = os.path.basename(dataset.imgs[i][0])
                save_path = os.path.join(target_dir, original_filename)
                
                # PIL 이미지로 저장 (ImageFolder로 다시 로드할 수 있도록)
                processed_img_pil.save(save_path)
            
            total_images += len(dataset)
            print(f"✓ [{split}] 전처리 완료: {len(dataset)}개 이미지 저장")
        
        print(f"\n{'='*50}")
        print(f"전체 전처리 완료!")
        print(f"  - 총 이미지: {total_images}개")
        print(f"  - 클래스: {sorted(all_classes)}")
        print(f"{'='*50}")
        
        # 5. 전처리된 결과를 새로운 Artifact로 업로드
        print(f"\n5. Artifact 업로드")
        
        # Transform 정보를 metadata로 저장
        transform_info = [t.__class__.__name__ for t in preprocessing_transform.transforms]
        
        processed_artifact = wandb.Artifact(
            name=cfg.artifact.processed_artifact_name,
            type=cfg.artifact.processed_artifact_type,
            description=cfg.artifact.description,
            metadata={
                "transforms": transform_info,
                "num_images": total_images,
                "classes": sorted(all_classes),
                "splits": splits,
                "raw_artifact": cfg.artifact.raw_artifact_name
            }
        )
        
        # 전처리된 디렉토리 추가
        processed_artifact.add_dir(processed_dir)
        
        # Artifact 로깅
        run.log_artifact(processed_artifact)
        
        print(f"✓ Artifact '{cfg.artifact.processed_artifact_name}' 업로드 완료")
        print(f"\n전처리 완료!")


if __name__ == '__main__':
    main()

