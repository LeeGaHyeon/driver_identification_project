'''
모델이 예측한 각 클래스에 대한 확률(softmax logits)을 계산하고, 각 실제 레이블에 대해 평균 확률을 계산한 후, csv로 저장하는 코드
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import os
import torch
import pandas as pd
import argparse 
from tqdm import tqdm
from dataset import *  
from model import * 
from utils.util import *
from main import *


'''
python save_logits_cross.py --kernel-type 0828_cross_w30 --data-folder images/ --enet-type Cross_model --CUDA_VISIBLE_DEVICES 7 # cross 모델 확률값 추출 코드
'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, default='256')
    parser.add_argument('--enet-type', type=str, required=True, default='tf_efficientnet_b0_ns')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    parser.add_argument('--out-dim', type=int, default=15)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='5')
    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='./prob')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    return args

def save_logits_to_csv(model, dataloader, csv_path, device, num_classes=15):
    model.eval()  
    all_probs = {i: [] for i in range(num_classes)}  
    
    try:
        batch = next(iter(dataloader))
        print('Batch loaded successfully!')
        # print('Batch loaded successfully:', batch)
    except Exception as e:
        print('Error loading batch:', e)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            try:
                print(f'Processing batch {i+1}')
                org_images, optical_images, labels = batch[:3]
                org_images = org_images.to(device)
                optical_images = optical_images.to(device)
                labels = labels.to(device)

                logits = model(org_images, optical_images, device)
                print(f'Logits for batch {i+1}: {logits}')  # 로그 출력으로 확인
                probs = torch.nn.functional.softmax(logits, dim=1)

                for j, label in enumerate(labels.cpu().numpy()):
                    all_probs[label].append(probs[j].cpu().numpy())
            except Exception as e:
                print(f'Error processing batch {i+1}: {e}')
                continue  # 오류가 발생한 경우 다음 배치로 넘어감

    # 평균 확률 계산
    avg_probs = {}
    for label, probs in all_probs.items():
        if probs:  # 빈 리스트가 아닌 경우에만 처리
            avg_probs[label] = np.mean(probs, axis=0)
        else:
            avg_probs[label] = np.zeros(num_classes)  # 비어 있는 경우 0으로 채우기

    # DataFrame으로 변환 (행: 실제 레이블, 열: 예측 레이블의 확률)
    df = pd.DataFrame.from_dict(avg_probs, orient='index', columns=[str(i) for i in range(num_classes)])

    # 레이블을 행과 열 이름으로 설정
    df.index.name = 'True Label'
    df.columns = [str(i) for i in df.columns]

    # CSV 파일로 저장
    df.to_csv(csv_path)
    print(f"Logits and labels saved to {csv_path}")

    
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True) 

    set_seed(4922)

    # 각 코스에 대한 logits 값을 저장할 경로
    course = ['A', 'B', 'C']
    count = ['1', '2', '3', '4']
    visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    optical = natsort.natsorted(glob.glob('./images/optical_window30/*****/****/***/**/*.jpg'))
    n = 30

    for j in range(len(course)):
        train_loader, valid_loader_set, valid_name = cross_main('course', course[j], visual, optical, n)

        # 모델 로드
        # model_file = os.path.join(args.model_dir, f'0902_AugAug2Cross_w30_best_fold_{valid_name}.pth.pth')
        model_file = os.path.join(args.model_dir, f'final_cross_w30_best_fold_{valid_name}.pth.pth')
        if os.path.isfile(model_file):
            model = Cross_model(num_frames=n, out_dim=args.out_dim, valid_name=valid_name)  # Cross_model로 변경
            model.load_state_dict(torch.load(model_file))
            model = model.to(device)
        else:
            print(f"Model file {model_file} not found.")
            continue

        # logits 저장 경로 설정
        csv_path = os.path.join(args.log_dir, f'{args.kernel_type}_logits_{valid_name}.csv')

        # logits 추출 및 저장
        save_logits_to_csv(model, valid_loader_set, csv_path, device)
        
    for j in range(len(count)):
        train_loader, valid_loader_set, valid_name = cross_main('count', count[j], visual, optical, n)

        # 모델 로드
        # model_file = os.path.join(args.model_dir, f'0902_AugAug2Cross_w30_best_fold_{valid_name}.pth.pth')
        model_file = os.path.join(args.model_dir, f'final_cross_w30_best_fold_{valid_name}.pth.pth')
        if os.path.isfile(model_file):
            model = Cross_model(num_frames=n, out_dim=args.out_dim, valid_name=valid_name)  # Cross_model로 변경
            model.load_state_dict(torch.load(model_file))
            model = model.to(device)
        else:
            print(f"Model file {model_file} not found.")
            continue

        # logits 저장 경로 설정
        csv_path = os.path.join(args.log_dir, f'{args.kernel_type}_logits_{valid_name}.csv')

        # logits 추출 및 저장
        save_logits_to_csv(model, valid_loader_set, csv_path, device)
        
if __name__ == '__main__':
    main()
