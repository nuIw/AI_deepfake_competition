import os
import subprocess
from torchvision.datasets import ImageFolder

LOCAL_DATA_PATH = '/content/data'
KAGGLE_JSON_DIR = '/content/drive/MyDrive/Colab Notebooks/kaggle'
KAGGLE_JSON_PATH = os.path.join(KAGGLE_JSON_DIR, 'kaggle.json')

def setup_kaggle_api():
    if not os.path.exists(KAGGLE_JSON_DIR):
        print('Kaggle JSON directory not found. Please download the Kaggle JSON file and place it in the kaggle directory.')
        return False
    
    print(f'Using Kaggle JSON file: {KAGGLE_JSON_PATH}')
    
    os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_JSON_DIR
    return True
    
def get_data_path():
    data_root = LOCAL_DATA_PATH
    print(f'Using local data path: {data_root}')
    
    return data_root

def download_kaggle_dataset(dataset_id, local_dir=LOCAL_DATA_PATH):
    if not setup_kaggle_api():
        print('Failed to setup Kaggle API. Please check the Kaggle JSON file and try again.')
        return
    
    print(f'Downloading dataset: {dataset_id} to {local_dir}')
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', dataset_id,
            '-p', local_dir,
            '--unzip',
        ], check=True)
        print(f'Dataset downloaded and extracted to {local_dir}')
    except Exception as e:
        print(f'Failed to download dataset: {e}')
        return

if __name__ == '__main__':
    #dataset 이름을 콘솔로 입력 받아서 다운로드 하는 코드 작성
    dataset_id = input('Enter the dataset ID \n it should be like this: "username/dataset_name": ')
    download_kaggle_dataset(dataset_id)
    
