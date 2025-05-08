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
- 학습 파일 run.py
python run.py --kernel-type 0801_self_VI --out-dim 15 --data-folder images/ --enet-type Self_VI --n-epochs 100 --batch-size 8 --k-fold 0 --image-size 224 --CUDA_VISIBLE_DEVICES 0
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
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--k-fold', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--accumulation-step', type=int, default=1)
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--use-ext', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)  # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=8)  # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=4e-6)  # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값 # 4e-5
    parser.add_argument('--n-epochs', type=int, default=200)  # epoch 수
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
def run(train_loader, valid_loader_set, valid_name, device):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss_list = []
    valid_loss_set_list = []
    acc_list = []
    if args.DEBUG:
        args.n_epochs = 5
        df = df.sample(args.batch_size * 3)

    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold_{valid_name}.pth')
    if 'MAE' == args.enet_type:
        ModelClass = MAE
    elif 'VIVIT' == args.enet_type:
        ModelClass = VIVIT
    elif 'Efficient' == args.enet_type:
        ModelClass = Efficient
    elif 'Cross_model' == args.enet_type:
        ModelClass = Cross_model
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

    else:
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            valid_name=valid_name
        )

    model = model.to(device)
    val_loss_max = 99999.
    val_loss_max2 = 99999.
    val_acc_max = float('-inf') 

    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)
    if DP:
        model = nn.DataParallel(model)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)

    classes = {'anseonyeong':0, 'baekseungdo':1, 'cheonaeji':2, 'choimingi':3, 'jeongyubin':4, 'jojeongduk':5, 'kimminju':6, 'leeeunseo':7, 'leegahyeon':8, 'leegihun':9, 'leejaeho':10, 'leekanghyuk':11, 'leeyunguel':12, 'seosanghyeok':13, 'simboseok':14}
    for epoch in range(1, args.n_epochs + 1):

        print(time.ctime(), f'Epoch {epoch}')
        
        train_loss,train_acc = single_train_epoch(model, train_loader, optimizer, device)
        val_loss_set, acc, confusion_pred, confusion_target, vote_3, vote_10, vote_all, logit_list, target_list = single_val_epoch(model,
                                                                                                   valid_loader_set,
                                                                                                   classes, device)

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid set1 loss: {(val_loss_set):.5f}, train_acc: {train_acc:.2f} val_acc: {(acc):.2f}, \n' \
                                       f'vote_3: {vote_3:.2f}, vote_10: {vote_10:.2f}, vote_all: {vote_all:.2f}'
        print(content)

        train_loss_list.append(train_loss)
        valid_loss_set_list.append(val_loss_set)
        acc_list.append(acc)
        
        confusion_matrix_str = f'Confusion Matrix:\n{confusion_matrix(confusion_target, confusion_pred)}'

        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_{valid_name}.txt'), 'a') as appender:
            appender.write(content + '\n')
            appender.write(confusion_matrix_str + '\n') 

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()  

        if val_loss_set < val_loss_max:
            print('val_loss_max1 ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set))
            torch.save(model.state_dict(), f"{model_file}_best_loss.pth")  # SR_weights
            val_loss_max = val_loss_set
            best_confusion_pred = confusion_pred
            best_confusion_target = confusion_target

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

            plt.subplot(3, 1, 2)
            plt.plot(acc_list)
            plt.legend(['val_acc'])
            plt.grid()

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

    precision = precision_score(best_confusion_target, best_confusion_pred, average='macro') 
    f1 = f1_score(best_confusion_target, best_confusion_pred, average='macro')

    top3_predicted = np.argsort(logit_list, axis=1)[:, -3:]

    correct_top3 = np.sum(np.equal(top3_predicted, target_list.reshape(-1, 1)))

    total_samples = len(target_list)

    top3_accuracy = correct_top3 / total_samples
    content3 = f'precision: {precision:.2f}, f1_score: {f1:.2f}, top-3: {top3_accuracy:.2f}'
    with open(os.path.join(args.log_dir, f'log_{args.kernel_type}_{valid_name}.txt'), 'a') as appender:
        appender.write(content3 + '\n')


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed(4922)
    device = torch.device('cuda')
    course = ['A','B','C']
    
    visual = natsort.natsorted(glob.glob('./images/visual_window30/*****/****/***/**/*.jpg'))
    optical = natsort.natsorted(glob.glob('./images/optical_window30/*****/****/***/**/*.jpg'))
    n=30
    
    for j in range(len(course)):
        train_loader, valid_loader_set, valid_name = fine_tuning_main('course', course[j], visual, n, mi=False)
    
        run(train_loader, valid_loader_set, valid_name, device)
