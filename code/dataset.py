# -*_ coding:utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
import time
import torchvision
# pip install git+https://github.com/hassony2/torch_videovision
from torchvideotransforms import video_transforms, volume_transforms
import albumentations
import random
import natsort
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T


# class종류
# classes = {'jojeongdeok': 0, 'leeyunguel': 1, 'huhongjune': 2, 'leegahyeon': 3,
#                'leejaeho': 4, 'leekanghyuk': 5, 'leeseunglee': 6, 'simboseok': 7, 'jeongyubin': 8, 'choimingi':9, 'leegihun': 10}

# excluded_leegihun
# classes = {'jojeongdeok': 0, 'leeyunguel': 1, 'huhongjune': 2, 'leegahyeon': 3,
#                'leejaeho': 4, 'leekanghyuk': 5, 'leeseunglee': 6, 'simboseok': 7, 'jeongyubin': 8, 'choimingi':9}

# total
# classes = {'seosanghyeok':0, 'ohseunghun':1, 'leeyunguel_2':2, 'leeseunglee':3, 'leekanghyuk_2':4, 'leejaeho_2':5, 'leegihun_2':6, 'leegahyeon_2':7, 'leeeunseo':8, 'kimminju':9, 'kimgangsu':10, 'kangminjae':11, 'kangjihyun':12, 'jojeongduk':13, 'hurhongjun':14, 'chunjihun':15, 'cheonaeji':16, 'baekseungdo':17, 'anseonyeong':18, 'leegihun':19, 'choimingi':20, 'jeongyubin':21, 'simboseok':22, 'leekanghyuk':23, 'leejaeho':24, 'leegahyeon':25, 'leeyunguel':26}

# classes = {'choimingi': 0, 'choimingi_auto': 0, 'houjonguk':1, 'houjonguk_auto':1, 'jeongyubin':2, 'jeongyubin_auto':2, 'leegahyeon':3, 'leegahyeon_auto':3}
event_classes = {'bump':0,'corner':1}

# auto_classes = {'choimingi': 1, 'choimingi_auto': 0, 'houjonguk':1, 'houjonguk_auto':0, 'jeongyubin':1, 'jeongyubin_auto':0, 'leegahyeon':1, 'leegahyeon_auto':0}
classes = {'choimingi': 0, 'houjonguk':1, 'jeongyubin':2, 'leegahyeon':3}


# class Finetuning_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
#     def __init__(self, video_path, mode='train', image_size=224, transform=None):
#         self.video_path = video_path
#         self.mode = mode
#         self.transform = transform
#         self.image_size = image_size
#         # self.video = natsort.natsorted(glob.glob('../data/visual/*****/****/***/**/*.avi'))

#     def __len__(self):
#         return len(self.video_path)

#     def __getitem__(self, idx): 
#         optical_path = self.video_path[idx]
#         id = optical_path[0].split('/')[-5]
#         event = optical_path[0].split('/')[-3]
#         event = event_classes[event]
#         label = classes[id]
#         optical_list = []
#         # auto_drive\data\visual_window_image\jojeongdeok\B\bump\1\img
#         #################### video
#         for pp in range(len(optical_path)):
#             optical = Image.open(optical_path[pp])
#             # optical = cv2.imread(optical_path[pp], cv2.IMREAD_COLOR)
#             # optical = cv2.cvtColor(optical, cv2.COLOR_BGR2RGB)

#             optical_list.append(optical)
#         optical_list = self.transform(optical_list).transpose(1, 0)

#         return optical_list.to(torch.float32), label, event


class Finetuning_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
    def __init__(self, video_path, mode='train', image_size=224, transform=None):
        self.video_path = video_path
        self.mode = mode
        self.transform = transform
        self.image_size = image_size 

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx): 
        optical_path = self.video_path[idx]
        id = optical_path[0].split('/')[3]
        label = classes[id]
        optical_list = [] 
        for pp in range(len(optical_path)):
            optical = Image.open(optical_path[pp]) 

            optical_list.append(optical)
        optical_list = self.transform(optical_list).transpose(1, 0)

        return optical_list.to(torch.float32), label



class Auto_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
    def __init__(self, video_path, optical_path, mode='train', image_size=224, transform=None, mi_transform=None):
        self.video_path = video_path
        self.optical_path = optical_path
        self.mode = mode
        self.transform = transform
        self.mi_transform = mi_transform
        self.image_size = image_size
        # self.video = natsort.natsorted(glob.glob('../data/visual/*****/****/***/**/*.avi'))
        self.csv = pd.read_csv('./auto_data/data_adas.csv')

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        # window 사용할 때
        # auto/window/name/1/img
        path = self.video_path[idx]
        optical_path = self.optical_path[idx]
        id = path[0].split('\\')[1]
        label = classes[id]

        video_path = path[0].split('\\')[-1].split('_mean')[0] + '.avi'
        auto_mode = int(np.array(self.csv[self.csv['video_path'] == video_path])[0,2]) # 0일 때 주행, 1일 때 보조모드
        # auto_mode
        # if auto_mode == 1:
        #     print(path[0])
        ####################
        visual_list = []
        optical_list = []

        for pp in range(len(path)):
            visual = Image.open(path[pp])
            visual_list.append(visual)  # cv2.WARP_FILL_OUTLIERS

            optical = Image.open(optical_path[pp])
            optical_list.append(optical)

        visual_list = self.transform(visual_list).transpose(1, 0)
        optical_list = self.mi_transform(optical_list).transpose(1, 0)

        return visual_list.to(torch.float32), optical_list.to(torch.float32), label, auto_mode




# './data/optical_npy/leeyunguel_optical_npy/C/corner/4/leeyunguel_C_c_4_13of14_0299.npy'
# class Cross_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
#     def __init__(self, video_path, optical_path, mode='train', image_size=224, transform=None, mi_transform=None):
#         self.video_path = video_path
#         self.optical_path = optical_path
#         self.mode = mode
#         self.transform = transform
#         self.mi_transform = mi_transform
#         self.image_size = image_size
#         # self.video = natsort.natsorted(glob.glob('../data/visual/*****/****/***/**/*.avi'))

#     def __len__(self):
#         return len(self.video_path)

#     def __getitem__(self, idx):
#         # window 사용할 때 
#         path = self.video_path[idx] 
#         optical_path = self.optical_path[idx]  
#         id = path[0].split('/')[-5]  
#         # print(id)
#         label = classes[id]
#         ####################
#         visual_list = []
#         optical_list = []

#         for pp in range(len(path)):
#             visual = Image.open(path[pp])
#             visual_list.append(visual)  # cv2.WARP_FILL_OUTLIERS

#             optical = Image.open(optical_path[pp])
#             optical_list.append(optical)

#         visual_list = self.transform(visual_list).transpose(1, 0)
#         optical_list = self.mi_transform(optical_list).transpose(1, 0)        
        
#         return visual_list.to(torch.float32), optical_list.to(torch.float32), label  # , #course

class Cross_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
    def __init__(self, video_path, optical_path, mode='train', image_size=224, transform=None, mi_transform=None):
        self.video_path = video_path
        self.optical_path = optical_path
        self.mode = mode
        self.transform = transform
        self.mi_transform = mi_transform
        self.image_size = image_size
        # self.video = natsort.natsorted(glob.glob('../data/visual/*****/****/***/**/*.avi'))

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        # window 사용할 때 
        path = self.video_path[idx] 
        optical_path = self.optical_path[idx]   
        id = path[0].split('/')[3]   
        label = classes[id] 
        ####################
        visual_list = []
        optical_list = []

        for pp in range(len(path)):
            visual = Image.open(path[pp])
            visual_list.append(visual)  # cv2.WARP_FILL_OUTLIERS

            optical = Image.open(optical_path[pp])
            optical_list.append(optical)

        visual_list = self.transform(visual_list).transpose(1, 0)
        optical_list = self.mi_transform(optical_list).transpose(1, 0)   
        
        return visual_list.to(torch.float32), optical_list.to(torch.float32), label  # , #course



class Sensor_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
    def __init__(self, video_path, raft_path, mode='train', image_size=224, transform=None, mi_transform=None):
        self.video_path = video_path
        self.raft_path = raft_path
        self.mode = mode
        self.transform = transform
        self.mi_transform = mi_transform
        self.image_size = image_size
        # self.video = natsort.natsorted(glob.glob('../data/visual/*****/****/***/**/*.avi'))

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        path = self.video_path[idx]
        id = path[0].split('\\')[1]
        label = classes[id]

        # pd.read_csv('./data/sensor_feature/leekanghyuk/A/3/leekanghyuk_A_3_Accel.csv')

        visual_list = []

        optical_path = self.raft_path[idx]
        optical_list = []

        for pp in range(len(path)):
            # visual = cv2.imread(path[pp], cv2.IMREAD_COLOR)
            # visual = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
            visual = Image.open(path[pp])
            visual_list.append(visual)  # cv2.WARP_FILL_OUTLIERS

            # optical = cv2.imread(optical_path[pp], cv2.IMREAD_COLOR)
            # optical = cv2.cvtColor(optical, cv2.COLOR_BGR2RGB)
            optical = Image.open(optical_path[pp])
            optical_list.append(optical)

        # visual_list = np.array(visual_list) # 10장
        visual_list = self.transform(visual_list).transpose(1, 0)
        optical_list = self.mi_transform(optical_list).transpose(1, 0)

        # sensor
        base_path = "/".join(path[0].split('\\')[1:-3])
        count = path[0].split('\\')[-2]
        sensor_path = glob.glob(os.path.join('./data/smartphone', base_path, count, '*.csv'))

        sensor = torch.from_numpy(np.array(pd.read_csv(sensor_path[0])))#[20:70][:,1:]#[random_num:random_num+30,:] # (226, 6)
        length = sensor.shape[0]
        random_num = random.randint(0, length-50)
        sensor = sensor[random_num:random_num+50,:][:,1:]
        # min_v, _ = torch.min(sensor, dim=0)
        # max_v, _ = torch.max(sensor, dim=0)
        # norm_sensor = (sensor - min_v) / (max_v- min_v)

        # 3169 x 9
        return visual_list.to(torch.float32), optical_list.to(torch.float32), sensor.to(torch.float32), label  # , #course
# def sensor_transfom():
#     transform = transforms.Com


class Video_CustomDataset(Dataset):  # train dataset 동적으로 만드는 class
    def __init__(self, video_path, mode='train', image_size=224, transform=None):
        self.video_path = video_path
        self.mode = mode
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        path = self.video_path[idx]
        id = path.split('\\')[1]
        label = classes[id]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {path}")

        frames = []
        num = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # if num >= 6:
            if num == 18:
                frame = cv2.resize(frame,dsize=(224,224))
                frame = Image.fromarray(frame)
                frames.append(frame)
                num=0
            num += 1
        cap.release()
        # frames = frames[6:]

        video_frames = self.transform(frames).transpose(1, 0)

        return video_frames.to(torch.float32), label
         
########################################
########################################
########################################
####### 기존 data aug ##################
########################################
########################################
########################################

# def get_transforms(img_size=224):
#     transforms_train = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(),
#         video_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
#         # video_transforms.RandomRotation(10),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_mi = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(),
#         # video_transforms.RandomRotation(10),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_val = video_transforms.Compose([
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     return transforms_train, transforms_mi, transforms_val

#######################################
#######################################
#######################################
###### 새로운 data aug (가현) ##########
#######################################
#######################################
#######################################

# def get_transforms(img_size=224):
#     transforms_train = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(),
#         video_transforms.RandomGrayscale(p=0.2),
#         video_transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0.25), 
#         video_transforms.RandomRotation(5),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_mi = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(), 
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_val = video_transforms.Compose([
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     return transforms_train, transforms_mi, transforms_val

########################################################
########################################################
########################################################
##### 새로운 data aug 2 (가현) - 최종 Augmentation ######
########################################################
########################################################
########################################################

def get_transforms(img_size=224):
    transforms_train = video_transforms.Compose([
        video_transforms.Resize(240),
        video_transforms.RandomCrop((224, 224)),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomGrayscale(p=0.2),
        video_transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0.25), 
        video_transforms.RandomRotation(5),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transforms_mi = video_transforms.Compose([
        video_transforms.Resize(240),
        video_transforms.CenterCrop((224, 224)),  # RandomCrop 대신 CenterCrop으로 단순화
        video_transforms.RandomRotation(10),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transforms_val = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return transforms_train, transforms_mi, transforms_val

###################################################################################
###################################################################################
# 새로운 data aug (아래에 코드 추가하고 위에 데이터 증강 부분 주석처리하고 돌리면 돼용) #
########### 코드 돌릴때 --kernel-type 에서 저장되는 모델명 지정해줘야해여 #############
###################################################################################
###################################################################################

########################################
########################################
########################################
####### 새로운 data aug (유빈) ##########
########################################
########################################
########################################
# def get_transforms(img_size=224):
#     transforms_train = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(),
#         video_transforms.RandomGrayscale(p=0.2),
#         video_transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.5, hue=0.25), 
#         video_transforms.RandomRotation(5),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_mi = video_transforms.Compose([
#         video_transforms.Resize(240),
#         video_transforms.RandomCrop((224, 224)),
#         video_transforms.RandomHorizontalFlip(), 
#         video_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     transforms_val = video_transforms.Compose([
#         volume_transforms.ClipToTensor(),
#         video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

#     return transforms_train, transforms_mi, transforms_val 