import os
import time
import argparse
from tqdm import tqdm
# import timm
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
from utils.sam import *
import pandas as pd
from dataset import *
from model import *
from train_def import *
from main import *
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
Precautions_msg = '(주의사항) ---- \n'
import torchsummary
import torch, torch.nn as nn, torch.nn.functional as F
import natsort
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

'''
- run_mi_girae_tobi.py

Training list
python run_concat.py --kernel-type 0823_concat_w30 --out-dim 15 --data-folder images/ --enet-type Concat_model --n-epochs 100 --batch-size 8 --k-fold 0 --image-size 224 --CUDA_VISIBLE_DEVICES 0
'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--data-dir', type=str, default='./data/')
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='256')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--enet-type', type=str, required=True, default='tf_efficientnet_b0_ns')
    # 학습에 적용할 네트워크 이름
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--use-meta', action='store_true')
    # meta데이터 (사진 외의 나이, 성별 등)을 사용할지 여부

    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    # meta데이터 사용 시 중간레이어 사이즈

    parser.add_argument('--out-dim', type=int, default=2)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')

    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    parser.add_argument('--k-fold', type=int, default=5)
    # data cross-validation
    # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/

    parser.add_argument('--accumulation-step', type=int, default=1)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    # parser.add_argument('--model-dir', type=str, default='./total_weights')
    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # 원본데이터에 추가로 외부 데이터를 사용할지 여부
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--batch-size', type=int, default=32)  # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=8)  # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=4e-6)  # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값 # 4e-5
    parser.add_argument('--n-epochs', type=int, default=200)  # epoch 수
    args, _ = parser.parse_known_args()
    return args

# def run(df, df_val, transforms_train, transforms_val, valid_name):

args = parse_args()
def run(train_loader, valid_loader_set, valid_name, device):
    # fold, df, transforms_train, transforms_val
    '''
    학습 진행 메인 함수
    :param fold: cross-validation에서 valid에 쓰일 분할 번호
    :param df: DataFrame 학습용 전체 데이터 목록
    :param transforms_train, transforms_val: 데이터셋 transform 함수
    '''
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss_list = []
    valid_loss_set_list = []
    acc_list = []
    if args.DEBUG:
        args.n_epochs = 5
        df = df.sample(args.batch_size * 3)

    # 학습된 weight 경로 있는지 확인
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold_{valid_name}.pth')
    # 학습된 weight 불러오기
    if 'MAE' == args.enet_type:
        ModelClass = MAE
    elif 'VIVIT' == args.enet_type:
        ModelClass = VIVIT
    elif 'Efficient' == args.enet_type:
        ModelClass = Efficient
    elif 'Cross_model' == args.enet_type:
        ModelClass = Cross_model
    elif 'Concat_model' == args.enet_type:
            ModelClass = Concat_model
    elif 'Self_VI' == args.enet_type:
        ModelClass = Self_VI
    elif 'Self_MI' == args.enet_type:
        ModelClass = Self_MI
    elif 'Cross_sensor' == args.enet_type:
        ModelClass = Cross_sensor
    elif 'LSTM' == args.enet_type:
        ModelClass = LSTM
    else:
        raise NotImplementedError()

    if os.path.isfile(model_file):
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            valid_name = valid_name
        )
        model.load_state_dict(torch.load(model_file))

        # loaded_state_dict = torch.load(model_file)
        # new_state_dict = OrderedDict()
        # for n, v in loaded_state_dict.items():
        #     name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
        #     new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
    else:
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            valid_name=valid_name
        )

    model = model.to(device)
    # print(torchsummary.summary(model, (10, 3, 224, 224)))
    # print(torchsummary.summary(model, ((10, 3, 224, 224), (10, 3, 224, 224))))
    val_loss_max = 99999.
    val_loss_max2 = 99999.
    val_acc_max = float('-inf')  # val_acc_max 변수 초기화

    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
    if DP:
        model = nn.DataParallel(model)

    # amp를 사용하면 버그 (use_amp 비활성화)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)

    # classes = {'jojeongdeok': 0, 'leeyunguel': 1, 'huhongjune': 2, 'leegahyeon': 3,
    #            'leejaeho': 4, 'leekanghyuk': 5, 'leeseunglee': 6, 'simboseok': 7, 'jeongyubin': 8, 'choimingi':9, 'leegihun': 10}
    # rot_class = {int(s * 10): idx for idx, s in enumerate(range(1, 11))}
    
    # classes = {'seosanghyeok':0, 'ohseunghun':1, 'leeyunguel_2':2, 'leeseunglee':3, 'leekanghyuk_2':4, 'leejaeho_2':5, 'leegihun_2':6, 'leegahyeon_2':7, 'leeeunseo':8, 'kimminju':9, 'kimgangsu':10, 'kangminjae':11, 'kangjihyun':12, 'jojeongduk':13, 'hurhongjun':14, 'chunjihun':15, 'cheonaeji':16, 'baekseungdo':17, 'anseonyeong':18, 'leegihun':19, 'choimingi':20, 'jeongyubin':21, 'simboseok':22, 'leekanghyuk':23, 'leejaeho':24, 'leegahyeon':25, 'leeyunguel':26}
 
    classes = {'anseonyeong':0, 'baekseungdo':1, 'cheonaeji':2, 'choimingi':3, 'jeongyubin':4, 'jojeongduk':5, 'kimminju':6, 'leeeunseo':7, 'leegahyeon':8, 'leegihun':9, 'leejaeho':10, 'leekanghyuk':11, 'leeyunguel':12, 'seosanghyeok':13, 'simboseok':14}
    for epoch in range(1, args.n_epochs + 1):

        print(time.ctime(), f'Epoch {epoch}')
        
        # train_loss,train_acc = single_train_epoch(model, train_loader, optimizer, device)
        # val_loss_set, acc, confusion_pred, confusion_target, vote_3, vote_10, vote_all, logit_list, target_list = single_val_epoch(model,
        #                                                                                            valid_loader_set,
        #                                                                                            classes, device)

        train_loss, train_acc = cross_train_epoch(model, train_loader, optimizer, device)
        val_loss_set, acc, confusion_pred, confusion_target, vote_3, vote_10, vote_all, logit_list, target_list = cross_val_epoch(model,
                                                                                                           valid_loader_set,
                                                                                                           classes, device)

        # train_loss, train_acc = video_train_epoch(model, train_loader, optimizer, device)
        # val_loss_set, acc, confusion_pred, confusion_target, vote_3, vote_10, vote_all, logit_list, target_list = video_val_epoch(model,
        #                                                                                                   valid_loader_set,
        #                                                                                                   classes,
        #                                                                                                   device)

        # train_loss, train_acc = cross_sensor_train_epoch(model, train_loader, optimizer, device)
        # val_loss_set, acc, confusion_pred, confusion_target, vote_3, vote_10, vote_all, logit_list, target_list = cross_sensor_val_epoch(model,
        #                                                                                                                                  valid_loader_set,
        #                                                                                                                                  classes, device)


        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid set1 loss: {(val_loss_set):.5f}, train_acc: {train_acc:.2f} val_acc: {(acc):.2f}, \n' \
                                       f'vote_3: {vote_3:.2f}, vote_10: {vote_10:.2f}, vote_all: {vote_all:.2f}'
        print(content)

        train_loss_list.append(train_loss)
        valid_loss_set_list.append(val_loss_set)
        acc_list.append(acc)
        
        confusion_matrix_str = f'Confusion Matrix:\n{confusion_matrix(confusion_target, confusion_pred)}'

        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_{valid_name}.txt'), 'a') as appender:
            appender.write(content + '\n')
            appender.write(confusion_matrix_str + '\n')  # confusion matrix 추가

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()  # bug workaround

        # if val_loss_set < val_loss_max:
        #     print('val_loss_max1 ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set))
        #     torch.save(model.state_dict(), model_file)  # SR_weights
        #     val_loss_max = val_loss_set
        #     best_confusion_pred = confusion_pred
        #     best_confusion_target = confusion_target
        
            # Loss가 가장 낮을 때 모델 저장
        if val_loss_set < val_loss_max:
            print('val_loss_max1 ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set))
            torch.save(model.state_dict(), f"{model_file}.pth")  # SR_weights
            val_loss_max = val_loss_set
            best_confusion_pred = confusion_pred
            best_confusion_target = confusion_target

        # Accuracy가 가장 높을 때 모델 저장
        # if acc > val_acc_max:
        #     print('val_acc_max ({:.2f} --> {:.2f}). Saving model ...'.format(val_acc_max, acc))
        #     torch.save(model.state_dict(), f"{model_file}_best_acc.pth")
        #     val_acc_max = acc

        # early_stopping
        early_stopping(val_loss_set, model)

        if early_stopping.early_stop or epoch == args.n_epochs:
            if early_stopping.early_stop:
                print("Early stopping epoch: ", epoch)
            plt.figure(figsize=(10, 30))
            plt.subplot(3, 1, 1)

            train_min = min(train_loss_list)
            train_x = np.argmin(train_loss_list)

            valid_min_set1 = min(valid_loss_set_list)
            valid_x_set1 = np.argmin(valid_loss_set_list)

            plt.plot(train_loss_list)
            plt.text(train_x, train_min, round(train_min, 4))
            plt.plot(valid_loss_set_list)
            plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1, 4))
            plt.legend(['train_loss', 'val_loss_set'])
            plt.ylabel('loss')
            plt.title(f'{args.kernel_type}')
            plt.grid()

            # 여기는 val_acc list
            plt.subplot(3, 1, 2)
            plt.plot(acc_list)
            plt.legend(['val_acc'])
            plt.grid()

            # # 여기다가 confusion matrix 나타내면 될 듯
            # plt.subplot(3, 1, 3)
            # plt.plot(valid_loss_set_list)
            # plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1, 4))
            # plt.legend(['val_loss_set'])
            # plt.grid()
            
            # 여기다가 confusion matrix 나타내면 될 듯
            plt.subplot(3, 1, 3)
            cm = confusion_matrix(best_confusion_target, best_confusion_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix')
            plt.grid()

            plt.savefig(f'./results/{args.kernel_type}_{valid_name}.jpg')
            # plt.show()
            break
    print('best_confusion_pred', best_confusion_pred)
    print('best_confusion_target', best_confusion_target)
    content2 = f'predict: {best_confusion_pred} \n' \
               f'target: {best_confusion_target}'
    with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_{valid_name}.txt'), 'a') as appender:
        appender.write(content2 + '\n')

    precision = precision_score(best_confusion_target, best_confusion_pred, average='macro')  # 'macro'는 다중 클래스의 평균을 사용하는 옵션입니다.
    f1 = f1_score(best_confusion_target, best_confusion_pred, average='macro')

    # 상위 3개 예측을 구함
    top3_predicted = np.argsort(logit_list, axis=1)[:, -3:]

    # 예측이 실제 레이블과 일치하는 개수를 셈
    correct_top3 = np.sum(np.equal(top3_predicted, target_list.reshape(-1, 1)))

    # 전체 데이터 개수
    total_samples = len(target_list)

    # 상위 3개 예측의 정확도 계산
    top3_accuracy = correct_top3 / total_samples
    content3 = f'precision: {precision:.2f}, f1_score: {f1:.2f}, top-3: {top3_accuracy:.2f}'
    with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_{valid_name}.txt'), 'a') as appender:
        appender.write(content3 + '\n')
    #################################################################


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    ###################################
    # 네트워크 타입 설정

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(4922)
    device = torch.device('cuda')
    course = ['A','B','C']
    count = ['1','2','3','4']

    #######################################
    #######################################
    #######################################

    # visual = natsort.natsorted(glob.glob('./images/visual_window10/*****/****/***/**/*.jpg'))
    # optical = natsort.natsorted(glob.glob('./images/optical_window10/*****/****/***/**/*.jpg'))
    # n = 10
    
    # visual = natsort.natsorted(glob.glob('./images/visual_window20/*****/****/***/**/*.jpg'))
    # optical = natsort.natsorted(glob.glob('./images/optical_window20/*****/****/***/**/*.jpg'))
    # n=20
    
    visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    optical = natsort.natsorted(glob.glob('./images/optical_window30/*****/****/***/**/*.jpg'))
    n=30
    
    # 제거 실험
    # visual = natsort.natsorted(glob.glob('/home/mmc/disk3/driver_identification/images/visual_window30/*****/****/***/**/*.jpg'))
    # optical = natsort.natsorted(glob.glob('/home/mmc/disk3/driver_identification/images/visual_window30/*****/****/***/**/*.jpg'))
    # n=30
    
    for j in range(len(count)):
        # train_loader, valid_loader_set, valid_name = fine_tuning_main('count', count[j], visual, n, mi=False)
        # train_loader, valid_loader_set, valid_name = fine_tuning_main('count', count[j], optical, n, mi=True) # mi 
    
        train_loader, valid_loader_set, valid_name = cross_main('count', count[j], visual, optical, n)
        # train_loader, valid_loader_set, valid_name = cross_sensor_main('count', count[j], visual, optical, n)
    
        run(train_loader, valid_loader_set, valid_name, device)
    
    for j in range(len(course)):
        # train_loader, valid_loader_set, valid_name = fine_tuning_main('course', course[j], visual, n, mi=False)
        # train_loader, valid_loader_set, valid_name = fine_tuning_main('course', course[j], optical, n, mi=True) # mi
    
        train_loader, valid_loader_set, valid_name = cross_main('course', course[j], visual, optical, n)
        # train_loader, valid_loader_set, valid_name = cross_sensor_main('course', course[j], visual, optical, n)
    
        run(train_loader, valid_loader_set, valid_name, device)

    #######################################
    #######################################
    #######################################
    
    # data_path = natsort.natsorted(glob.glob('./data/visual/*****/****/***/**/*.avi'))
    # print(data_path)
    # #
    # # for j in range(len(count)):
    # #     train_loader, valid_loader_set, valid_name = video_main('count', count[j], data_path)
    # #     run(train_loader, valid_loader_set, valid_name, device)
    
    # for j in range(len(course)):
    #     train_loader, valid_loader_set, valid_name = video_main('course', course[j], data_path)
    #     run(train_loader, valid_loader_set, valid_name, device)

    # #######################################
    # #######################################
    # #######################################

# if __name__ == '__main__': 
    
#     # argument값 만들기
#     args = parse_args()
#     os.makedirs(args.model_dir, exist_ok=True)
#     os.makedirs(args.log_dir, exist_ok=True)
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.CUDA_VISIBLE_DEVICES)

#     # 네트워크 타입 설정
#     DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

#     # 실험 재현을 위한 random seed 부여하기
#     set_seed(4922)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     course = ['A', 'B', 'C']
#     count = ['1', '2', '3', '4']

#     # 데이터 경로 설정
#     visual_path = os.path.join(args.data_folder, 'visual_window10') # images/visual_window10
#     optical_path = os.path.join(args.data_folder, 'optical_window10') # images/optical_window10 

#     for j in range(len(count)): 
#         train_dataset = Cross_CustomDataset(video_path=visual_path, optical_path=optical_path, mode='train', image_size=args.image_size)
#         valid_dataset = Cross_CustomDataset(video_path=visual_path, optical_path=optical_path, mode='valid', image_size=args.image_size)
        
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        
#         run(train_loader, [valid_loader], 'valid_name', device)

#     for j in range(len(course)):
#         train_dataset = Cross_CustomDataset(video_path=visual_path, optical_path=optical_path, mode='train', image_size=args.image_size)
#         valid_dataset = Cross_CustomDataset(video_path=visual_path, optical_path=optical_path, mode='valid', image_size=args.image_size)
        
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        
#         run(train_loader, [valid_loader], 'valid_name', device)