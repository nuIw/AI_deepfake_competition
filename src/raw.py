"""
wandb Artifact 관리 모듈

이 모듈은 wandb Artifact를 사용하여 데이터셋을 관리하는 기능을 제공합니다.
"""

import os
import wandb
from pathlib import Path
from typing import Optional


class ArtifactManager:
    """wandb Artifact를 관리하는 클래스"""
    
    def __init__(self, project_name: str, entity: Optional[str] = None):
        """
        ArtifactManager 초기화
        
        Args:
            project_name: wandb 프로젝트 이름
            entity: wandb entity (팀 또는 사용자 이름)
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
    
    def _ensure_wandb_init(self):
        """wandb가 초기화되지 않았으면 초기화"""
        if not wandb.run:
            self.run = wandb.init(project=self.project_name, entity=self.entity, job_type="artifact_management")
        else:
            self.run = wandb.run
    
    def create_artifact(
        self, 
        artifact_name: str, 
        artifact_type: str,
        directory_path: str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> wandb.Artifact:
        """
        1. 경로에 있는 디렉토리를 아티팩트에 넣어 저장하는 기능
        
        Args:
            artifact_name: 생성할 아티팩트 이름 (예: "dataset", "processed-data")
            artifact_type: 아티팩트 타입 (예: "dataset", "model", "result")
            directory_path: 아티팩트에 추가할 디렉토리 경로
            description: 아티팩트 설명 (선택)
            metadata: 아티팩트 메타데이터 (선택)
        
        Returns:
            생성된 wandb.Artifact 객체
        
        Example:
            >>> manager = ArtifactManager(project_name="my-project")
            >>> artifact = manager.create_artifact(
            ...     artifact_name="my-dataset",
            ...     artifact_type="dataset",
            ...     directory_path="./data/raw",
            ...     description="Raw dataset for training"
            ... )
        """
        self._ensure_wandb_init()
        
        # 디렉토리 존재 확인
        if not os.path.exists(directory_path):
            raise ValueError(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"경로가 디렉토리가 아닙니다: {directory_path}")
        
        # 아티팩트 생성
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )
        
        # 디렉토리 추가
        artifact.add_dir(directory_path)
        
        # 아티팩트 로깅 (업로드)
        self.run.log_artifact(artifact)
        
        print(f"✓ 아티팩트 '{artifact_name}' 생성 완료")
        print(f"  - 타입: {artifact_type}")
        print(f"  - 경로: {directory_path}")
        
        return artifact
    
    def extend_artifact(
        self,
        base_artifact_name: str,
        new_artifact_name: str,
        directory_path: str,
        artifact_type: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> wandb.Artifact:
        """
        2. 기존의 아티팩트를 가져오고 경로에 있는 디렉토리를 추가해서 새로운 아티팩트로 저장하는 기능
        
        Args:
            base_artifact_name: 기존 아티팩트 이름 (예: "my-dataset:v0" 또는 "my-dataset:latest")
            new_artifact_name: 새로운 아티팩트 이름
            directory_path: 추가할 디렉토리 경로
            artifact_type: 새 아티팩트 타입 (None이면 기존 아티팩트와 동일)
            description: 새 아티팩트 설명 (선택)
            metadata: 새 아티팩트 메타데이터 (선택)
        
        Returns:
            새로운 wandb.Artifact 객체
        
        Example:
            >>> manager = ArtifactManager(project_name="my-project")
            >>> artifact = manager.extend_artifact(
            ...     base_artifact_name="my-dataset:latest",
            ...     new_artifact_name="my-dataset-extended",
            ...     directory_path="./data/additional",
            ...     description="Extended dataset with additional data"
            ... )
        """
        self._ensure_wandb_init()
        
        # 디렉토리 존재 확인
        if not os.path.exists(directory_path):
            raise ValueError(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"경로가 디렉토리가 아닙니다: {directory_path}")
        
        # 기존 아티팩트 다운로드
        print(f"기존 아티팩트 '{base_artifact_name}' 다운로드 중...")
        base_artifact = self.run.use_artifact(base_artifact_name)
        
        # 임시 디렉토리에 다운로드
        base_artifact_dir = base_artifact.download()
        print(f"✓ 기존 아티팩트 다운로드 완료: {base_artifact_dir}")
        
        # 아티팩트 타입 결정
        if artifact_type is None:
            artifact_type = base_artifact.type
        
        # 새로운 아티팩트 생성
        new_artifact = wandb.Artifact(
            name=new_artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )
        
        # 기존 아티팩트 내용 추가
        print("기존 아티팩트 내용 추가 중...")
        new_artifact.add_dir(base_artifact_dir)
        
        # 새로운 디렉토리 추가
        print(f"새로운 디렉토리 '{directory_path}' 추가 중...")
        new_artifact.add_dir(directory_path)
        
        # 새로운 아티팩트 로깅 (업로드)
        self.run.log_artifact(new_artifact)
        
        print(f"✓ 확장된 아티팩트 '{new_artifact_name}' 생성 완료")
        print(f"  - 타입: {artifact_type}")
        print(f"  - 기존 아티팩트: {base_artifact_name}")
        print(f"  - 추가된 경로: {directory_path}")
        
        return new_artifact
    
    def download_artifact(
        self,
        artifact_name: str,
        download_path: str,
        aliases: Optional[list] = None
    ) -> str:
        """
        3. 기존의 아티팩트를 입력받은 경로에 저장하는 기능
        
        Args:
            artifact_name: 다운로드할 아티팩트 이름 (예: "my-dataset:v0" 또는 "my-dataset:latest")
            download_path: 아티팩트를 저장할 경로
            aliases: 아티팩트에 붙일 별칭 리스트 (선택, 예: ["latest", "production"])
        
        Returns:
            아티팩트가 저장된 경로
        
        Example:
            >>> manager = ArtifactManager(project_name="my-project")
            >>> saved_path = manager.download_artifact(
            ...     artifact_name="my-dataset:latest",
            ...     download_path="./data/downloaded"
            ... )
        """
        self._ensure_wandb_init()
        
        # 다운로드 경로 생성
        os.makedirs(download_path, exist_ok=True)
        
        # 아티팩트 다운로드
        print(f"아티팩트 '{artifact_name}' 다운로드 중...")
        artifact = self.run.use_artifact(artifact_name)
        
        # 지정된 경로로 다운로드
        artifact_dir = artifact.download(root=download_path)
        
        print(f"✓ 아티팩트 다운로드 완료")
        print(f"  - 아티팩트: {artifact_name}")
        print(f"  - 저장 경로: {artifact_dir}")
        print(f"  - 버전: {artifact.version}")
        print(f"  - 타입: {artifact.type}")
        
        return artifact_dir
    
    def finish(self):
        """wandb run 종료"""
        if self.run:
            self.run.finish()
            print("✓ wandb run 종료")


def main():
    """
    사용 예제 및 CLI 인터페이스
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="wandb Artifact 관리 도구")
    parser.add_argument("--project", type=str, required=True, help="wandb 프로젝트 이름")
    parser.add_argument("--entity", type=str, default=None, help="wandb entity (선택)")
    
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령")
    
    # create 명령
    create_parser = subparsers.add_parser("create", help="새 아티팩트 생성")
    create_parser.add_argument("--name", type=str, required=True, help="아티팩트 이름")
    create_parser.add_argument("--type", type=str, required=True, help="아티팩트 타입")
    create_parser.add_argument("--dir", type=str, required=True, help="추가할 디렉토리 경로")
    create_parser.add_argument("--description", type=str, default=None, help="아티팩트 설명")
    
    # extend 명령
    extend_parser = subparsers.add_parser("extend", help="기존 아티팩트 확장")
    extend_parser.add_argument("--base", type=str, required=True, help="기존 아티팩트 이름")
    extend_parser.add_argument("--name", type=str, required=True, help="새 아티팩트 이름")
    extend_parser.add_argument("--dir", type=str, required=True, help="추가할 디렉토리 경로")
    extend_parser.add_argument("--type", type=str, default=None, help="아티팩트 타입")
    extend_parser.add_argument("--description", type=str, default=None, help="아티팩트 설명")
    
    # download 명령
    download_parser = subparsers.add_parser("download", help="아티팩트 다운로드")
    download_parser.add_argument("--name", type=str, required=True, help="아티팩트 이름")
    download_parser.add_argument("--path", type=str, required=True, help="저장할 경로")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # ArtifactManager 초기화
    manager = ArtifactManager(project_name=args.project, entity=args.entity)
    
    try:
        if args.command == "create":
            manager.create_artifact(
                artifact_name=args.name,
                artifact_type=args.type,
                directory_path=args.dir,
                description=args.description
            )
        
        elif args.command == "extend":
            manager.extend_artifact(
                base_artifact_name=args.base,
                new_artifact_name=args.name,
                directory_path=args.dir,
                artifact_type=args.type,
                description=args.description
            )
        
        elif args.command == "download":
            manager.download_artifact(
                artifact_name=args.name,
                download_path=args.path
            )
    
    finally:
        manager.finish()


if __name__ == "__main__":
    main()

