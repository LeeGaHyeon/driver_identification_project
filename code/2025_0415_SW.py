#!/usr/bin/env python3
"""
driver_id_modality.py

이 스크립트는 단일 모달리티(영상, 센서, 또는 CAN 데이터)를 이용한 운전자 식별을 구현합니다.
사용자는 아래의 모달리티 중 하나를 선택할 수 있습니다.
    • "video"   : 블랙박스 영상 데이터를 이용 (예: Finetuning_CustomDataset 사용)
    • "sensor"  : 스마트폰 센서 데이터를 이용 (예: Sensor_CustomDataset 사용)
    • "can"     : CAN 데이터를 이용 (CAN 전용 데이터셋 클래스로 수정 필요)

선택한 모달리티에 따라 해당 데이터셋을 로드하고, 모델 파이프라인(Efficient 모델 예제)을 초기화하여 학습을 진행합니다.
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 데이터셋 클래스 임포트
# 영상 데이터 처리를 위한 Finetuning_CustomDataset
# 센서 데이터 처리를 위한 Sensor_CustomDataset
from dataset import Finetuning_CustomDataset, Sensor_CustomDataset
# CAN 데이터의 경우, 예제에서는 영상 데이터셋을 재활용합니다.
# 실제 사용 시 CAN 데이터 전용 데이터셋 클래스로 수정하세요.
from dataset import Finetuning_CustomDataset as CAN_CustomDataset

# 모델 임포트 (Efficient 모델 예제)
from model import Efficient

# 이미지 transform 함수 임포트 (데이터 전처리용)
from dataset import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="단일 모달리티 기반 운전자 식별")
    parser.add_argument('--modality', type=str, required=True, choices=['video', 'sensor', 'can'],
                        help="사용할 모달리티 선택: video, sensor, 또는 can")
    parser.add_argument('--data-folder', type=str, required=True,
                        help="모달리티별 데이터가 저장된 폴더 경로")
    parser.add_argument('--image-size', type=int, default=224,
                        help="입력 이미지 크기 (해당되는 경우)")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="학습 배치 사이즈")
    parser.add_argument('--n-epochs', type=int, default=100,
                        help="학습 에포크 수")
    parser.add_argument('--learning-rate', type=float, default=4e-6,
                        help="초기 학습률")
    parser.add_argument('--model-dir', type=str, default='./weights',
                        help="가중치를 저장하고 로드할 디렉토리")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="학습에 사용할 디바이스 (cuda 또는 cpu)")
    return parser.parse_args()


def get_data_loaders(modality, data_folder, image_size, batch_size):
    """
    선택된 모달리티에 따라 학습 및 검증 DataLoader를 생성합니다.
    (여기서는 데이터 폴더 내 파일을 단순하게 읽어 80/20 비율로 분할하는 예제를 사용합니다.
     실제 환경에 맞게 수정하여 사용하세요.)
    """
    all_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder)
                        if os.path.isfile(os.path.join(data_folder, f))])
    
    n_total = len(all_files)
    n_train = int(n_total * 0.8)
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]
    
    # 이미지 전처리(transform) 함수를 불러옵니다.
    transforms_train, transforms_mi, transforms_val = get_transforms(image_size)
    
    if modality == 'video':
        # 영상 데이터의 경우, 연속된 이미지들을 한 클립으로 묶습니다.
        N = 10  # 한 샘플 당 프레임 수 (필요에 따라 조정)
        train_paths = [train_files[i:i+N] for i in range(0, len(train_files), N) if len(train_files[i:i+N]) == N]
        val_paths   = [val_files[i:i+N] for i in range(0, len(val_files), N) if len(val_files[i:i+N]) == N]
        dataset_train = Finetuning_CustomDataset(train_paths, mode='train', image_size=image_size, transform=transforms_mi)
        dataset_val   = Finetuning_CustomDataset(val_paths, mode='val', image_size=image_size, transform=transforms_mi)
    elif modality == 'sensor':
        # 센서 데이터의 경우, Sensor_CustomDataset을 이용하여 데이터를 로드합니다.
        dataset_train = Sensor_CustomDataset(train_files, train_files, mode='train', image_size=image_size,
                                               transform=transforms_mi, mi_transform=transforms_mi)
        dataset_val   = Sensor_CustomDataset(val_files, val_files, mode='val', image_size=image_size,
                                               transform=transforms_mi, mi_transform=transforms_mi)
    elif modality == 'can':
        # CAN 데이터의 경우, 예제에서는 영상 데이터셋을 재사용합니다.
        # 실제로는 CAN 데이터 전용 데이터셋 클래스로 대체하세요.
        dataset_train = CAN_CustomDataset(train_files, mode='train', image_size=image_size, transform=transforms_mi)
        dataset_val   = CAN_CustomDataset(val_files, mode='val', image_size=image_size, transform=transforms_mi)
    else:
        raise ValueError("지원되지 않는 모달리티가 선택되었습니다.")
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in data_loader:
        # 각 배치는 (입력, 라벨) 튜플로 구성되어 있다고 가정합니다.
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(data_loader)
    return average_loss


def validate(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    average_loss = running_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    print(f"선택된 모달리티: {args.modality}")
    print(f"데이터 폴더: {args.data_folder}")
    train_loader, val_loader = get_data_loaders(args.modality, args.data_folder, args.image_size, args.batch_size)
    
    # 모델 초기화 (여기서는 영상 데이터를 위한 Efficient 모델 사용 예제)
    num_classes = 4  # 운전자 클래스 수에 맞게 조정하세요.
    model = Efficient(enet_type='Efficient', out_dim=num_classes)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"{time.ctime()} 에포크 {epoch}: train loss = {train_loss:.5f}, val loss = {val_loss:.5f}, "
              f"val accuracy = {val_acc * 100:.2f}%")
        
        # 검증 손실이 낮으면 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.model_dir, f"{args.modality}_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"최적 모델 저장: {model_path}")


if __name__ == "__main__":
    main()
