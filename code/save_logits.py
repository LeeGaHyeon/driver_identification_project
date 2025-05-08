'''
모델이 예측한 각 클래스에 대한 확률(softmax logits)을 계산하고, 각 실제 레이블에 대해 평균 확률을 계산한 후, csv로 저장하는 코드
'''
import os
import torch
import pandas as pd
import argparse  # argparse 모듈을 import
from tqdm import tqdm
from dataset import *  # 적절한 Dataset import 필요
from model import *  # 적절한 Model import 필요
from utils.util import *
from main import fine_tuning_main

'''
python save_logits.py --kernel-type 0801_fine_VI --data-folder images/ --enet-type Efficient --CUDA_VISIBLE_DEVICES 0 # vi 모델 확률값 추출 코드
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
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    return args
 
def save_logits_to_csv(model, dataloader, csv_path, device, num_classes=15):
    model.eval() # 모델 평가 모드로 전환
    all_probs = {i: [] for i in range(num_classes)}  # 각 레이블별로 확률 저장할 딕셔너리 {0: [], 1: [], 2: [], ..., 14: []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels = batch[:2]  # 첫 번째 값 (이미지)와 두 번째 값 (라벨)만 사용
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            logits = logits[0]  # tuple의 첫 번째 요소에 접근
            probs = F.softmax(logits, dim=1)  # 확률로 변환
            
            # 각 레이블에 해당하는 확률을 딕셔너리에 추가
            for i, label in enumerate(labels.cpu().numpy()):
                all_probs[label].append(probs[i].cpu().numpy())
    
    # 평균 확률 계산
    avg_probs = {label: np.mean(probs, axis=0) for label, probs in all_probs.items()}
    
    # DataFrame으로 변환 (행: 실제 레이블, 열: 예측 레이블의 확률)
    df = pd.DataFrame.from_dict(avg_probs, orient='index', columns=[str(i) for i in range(num_classes)])
    
    # 레이블을 행과 열 이름으로 설정
    df.index.name = 'True Label'
    # df.columns = ['Predicted_' + str(i) for i in df.columns]
    df.columns = [str(i) for i in df.columns]
    
    # CSV 파일로 저장
    df.to_csv(csv_path)
    print(f"Logits and labels saved to {csv_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed(4922)

    # 각 코스에 대한 logits 값을 저장할 경로
    course = ['A', 'B', 'C']
    visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    n = 30

    for j in range(len(course)):
        train_loader, valid_loader_set, valid_name = fine_tuning_main('course', course[j], visual, n, mi=False)

        # 모델 로드
        model_file = os.path.join(args.model_dir, f'0801_fine_VI_w30_best_fold_{valid_name}.pth_best_loss.pth')
        if os.path.isfile(model_file):
            model = Efficient(args.enet_type, out_dim=args.out_dim, valid_name=valid_name)  # 적절한 ModelClass 사용
            model.load_state_dict(torch.load(model_file))
            model = model.to(device)
        else:
            print(f"Model file {model_file} not found.")
            continue

        # logits 저장 경로 설정
        csv_path = os.path.join(args.log_dir, f'{args.kernel_type}_logits_{valid_name}.csv')

        # logits 추출 및 저장
        save_logits_to_csv(model, valid_loader_set, csv_path, device)
    
    # # 각 회차에 대한 logits 값을 저장할 경로
    # count = ['1', '2', '3', '4']
    # visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    # n = 30

    # for j in range(len(count)):
    #     train_loader, valid_loader_set, valid_name = fine_tuning_main('count', count[j], visual, n, mi=False)

    #     # 모델 로드
    #     model_file = os.path.join(args.model_dir, f'0801_fine_VI_w30_best_fold_{valid_name}.pth_best_loss.pth')
    #     if os.path.isfile(model_file):
    #         model = Efficient(args.enet_type, out_dim=args.out_dim, valid_name=valid_name)  # 적절한 ModelClass 사용
    #         model.load_state_dict(torch.load(model_file))
    #         model = model.to(device)
    #     else:
    #         print(f"Model file {model_file} not found.")
    #         continue

    #     # logits 저장 경로 설정
    #     csv_path = os.path.join(args.log_dir, f'{args.kernel_type}_logits_{valid_name}.csv')

    #     # logits 추출 및 저장
    #     save_logits_to_csv(model, valid_loader_set, csv_path, device)

if __name__ == '__main__':
    main()
