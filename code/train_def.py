import os
import time
from tqdm import tqdm
# import timm
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
from utils.sam import *
# import apex
# from apex import amp


from dataset import *
from model import *
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
Precautions_msg = '(주의사항) ---- \n'
import torchsummary
import torch, torch.nn as nn, torch.nn.functional as F
import natsort


criterion = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()
def single_train_epoch(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, target) in enumerate(bar):
        optimizer.zero_grad()
        org_data, target = org_data.to(device), target.to(device).long()
        # original data + motion data를 model에 매개변수로
        predict = model(org_data)

        loss = criterion(predict, target)
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        ps = F.softmax(predict, dim=1)  # model logit을 확률로

        top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)

    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def single_val_epoch(model, loader, rot_class, device):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = []
    with torch.no_grad():
        for (org_data, target) in tqdm(loader):
            org_data, target = org_data.to(device), target.to(device).long()
            predict = model(org_data)

            #single efficient
            # predict = model(org_data)

            loss = criterion(predict.to(torch.float32), target)
            val_loss.append(loss.detach().cpu().numpy())

            ps = F.softmax(predict, dim=1)  # model logit을 확률로
            top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100

    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0
    ##############################
    ############################## voting
    ##############################
    for i in range(len(ps[0])):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += np.argmax(softmax) == i
        vote_all_t += 1
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
            voting_3 += np.argmax(softmax_3) == i
            voting_3_t += 1
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
            voting_10 += np.argmax(softmax_10) == i
            voting_10_t += 1
    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3/voting_3_t)*100, (voting_10/voting_10_t)*100, (vote_all/vote_all_t)*100, np.array(logit_list), np.array(target_list)


#######################################
#######################################
#######################################
def auto_train_epoch(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, mi_data, target, auto_label) in enumerate(bar):
        optimizer.zero_grad()
        org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()
        auto_label = auto_label.to(device).unsqueeze(dim=1).to(torch.float32)

        # single efficient
        # predict = model(org_data)
        predict, auto_pred = model(org_data, mi_data, device)

        loss = criterion(predict, target)
        loss_event = criterion_bce(auto_pred, auto_label)
        loss = loss + loss_event
        # original data + motion data를 model에 매개변수로

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        ps = F.softmax(predict, dim=1)  # model logit을 확률로

        top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)

    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def auto_val_epoch(model, loader, rot_class, device):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = []
    with torch.no_grad():
        for (org_data, mi_data, target, auto_label) in tqdm(loader):
            org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()
            auto_label = auto_label.to(device).unsqueeze(dim=1).to(torch.float32)

            # single efficient
            # predict = model(org_data)
            predict, auto_pred = model(org_data, mi_data, device)

            loss = criterion(predict, target)
            loss_event = criterion_bce(auto_pred, auto_label)
            loss = loss + loss_event

            val_loss.append(loss.detach().cpu().numpy())

            ps = F.softmax(predict, dim=1)  # model logit을 확률로
            top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100
    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0

    pred_all = []
    pred_3 = []
    pred_10 = []
    target_all = []
    target_3 = []
    target_10 = []
    ##############################
    ############################## voting
    ##############################len(ps[0])
    # print(ps)
    # print(target_list)
    # if event == 'bump':
    for i in range(len(ps[0])):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += np.argmax(softmax) == i
        vote_all_t += 1

        pred_all.append(softmax)
        target_all.append(i)
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
            voting_3 += np.argmax(softmax_3) == i
            voting_3_t += 1

            pred_3.append(softmax_3)
            target_3.append(i)
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
            voting_10 += np.argmax(softmax_10) == i
            voting_10_t += 1

            pred_10.append(softmax_10)
            target_10.append(i)

    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3 / voting_3_t) * 100, (
            voting_10 / voting_10_t) * 100, (vote_all / vote_all_t) * 100, np.array(logit_list), np.array(
        target_list), np.array(pred_3), np.array(target_3), np.array(pred_10), np.array(target_10), np.array(
        pred_all), np.array(target_all)




#######################################
#######################################
#######################################
#######################################


def cross_train_epoch(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, mi_data, target) in enumerate(bar):
        optimizer.zero_grad()
        org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()

        # original data + motion data를 model에 매개변수로
        predict = model(org_data, mi_data, device)

        loss = criterion(predict, target)
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        ps = F.softmax(predict, dim=1)  # model logit을 확률로

        top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)

    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def cross_val_epoch(model, loader, rot_class, device): 
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))
     

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = [] 
     
    with torch.no_grad():
        for (org_data, mi_data, target) in tqdm(loader): 
            print('접속')
            org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()

            predict = model(org_data, mi_data, device)
            loss = criterion(predict, target)

            val_loss.append(loss.detach().cpu().numpy())

            ps = F.softmax(predict, dim=1)  # model logit을 확률로
            top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100
    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0
    ##############################
    ############################## voting
    ##############################len(ps[0])
    for i in range(len(ps[0])):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += np.argmax(softmax) == i
        vote_all_t += 1
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
            voting_3 += np.argmax(softmax_3) == i
            voting_3_t += 1
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
            voting_10 += np.argmax(softmax_10) == i
            voting_10_t += 1
    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3 / voting_3_t) * 100, (
                voting_10 / voting_10_t) * 100, (vote_all / vote_all_t) * 100, np.array(logit_list), np.array(target_list)


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################


def video_train_epoch(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, target) in enumerate(bar):
        optimizer.zero_grad()
        org_data, target = org_data.to(device), target.to(device).long()
        # event = event.to(device).unsqueeze(dim=1).to(torch.float32)
        # original data + motion data를 model에 매개변수로
        predict = model(org_data)

        loss = criterion(predict, target)
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        ps = F.softmax(predict, dim=1)  # model logit을 확률로

        top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)

    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def video_val_epoch(model, loader, rot_class, device):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = []
    with torch.no_grad():
        for (org_data, target) in tqdm(loader):
            org_data, target = org_data.to(device), target.to(device).long()
            predict = model(org_data)

            #single efficient
            # predict = model(org_data)

            loss = criterion(predict.to(torch.float32), target)

            val_loss.append(loss.detach().cpu().numpy())

            ps = F.softmax(predict, dim=1)  # model logit을 확률로
            top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100
    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0

    pred_all = []
    pred_3 = []
    pred_10 = []
    target_all = []
    target_3 = []
    target_10 = []
    ##############################
    ############################## voting
    ##############################len(ps[0])
    # print(ps)
    # print(target_list)
    # if event == 'bump':
    for i in range(len(ps[0])):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += np.argmax(softmax) == i
        vote_all_t += 1

        pred_all.append(softmax)
        target_all.append(i)
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
            voting_3 += np.argmax(softmax_3) == i
            voting_3_t += 1

            pred_3.append(softmax_3)
            target_3.append(i)
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
            voting_10 += np.argmax(softmax_10) == i
            voting_10_t += 1

            pred_10.append(softmax_10)
            target_10.append(i)

    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3 / voting_3_t) * 100, (
            voting_10 / voting_10_t) * 100, (vote_all / vote_all_t) * 100, np.array(logit_list), np.array(
        target_list), np.array(pred_3), np.array(target_3), np.array(pred_10), np.array(target_10), np.array(
        pred_all), np.array(target_all)


def cross_sensor_train_epoch(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, mi_data, sensor, target) in enumerate(bar):
        optimizer.zero_grad()
        org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()
        sensor = sensor.to(device)
        # original data + motion data를 model에 매개변수로
        predict = model(org_data, mi_data, device,sensor)

        loss = criterion(predict, target)
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

        ps = F.softmax(predict, dim=1)  # model logit을 확률로

        top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)

    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def cross_sensor_val_epoch(model, loader, rot_class, device):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = []
    with torch.no_grad():
        for (org_data, mi_data, sensor, target) in tqdm(loader):
            org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()
            sensor = sensor.to(device)
            predict = model(org_data, mi_data, device,sensor)
            loss = criterion(predict, target)

            val_loss.append(loss.detach().cpu().numpy())

            ps = F.softmax(predict, dim=1)  # model logit을 확률로
            top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100
    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0
    ##############################
    ############################## voting
    ##############################len(ps[0])
    for i in range(len(ps[0])):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += np.argmax(softmax) == i
        vote_all_t += 1
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
            voting_3 += np.argmax(softmax_3) == i
            voting_3_t += 1
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
            voting_10 += np.argmax(softmax_10) == i
            voting_10_t += 1
    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3 / voting_3_t) * 100, (
                voting_10 / voting_10_t) * 100, (vote_all / vote_all_t) * 100, np.array(logit_list), np.array(target_list)


def auto_train_epoch2(model, loader, optimizer, device):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    correct = 0
    total = 0

    for i, (org_data, mi_data, target, auto_label) in enumerate(bar):
        optimizer.zero_grad()
        org_data, mi_data, target = org_data.to(device), mi_data.to(device), target.to(device).long()
        auto_label = auto_label.to(device).unsqueeze(dim=1).to(torch.float32)

        # single efficient
        # predict = model(org_data)
        predict, auto_pred = model(org_data, mi_data, device)

        loss = criterion_bce(auto_pred, auto_label)
        # loss = criterion_bce(auto_pred, auto_label)
        # loss = loss + loss_event
        # original data + motion data를 model에 매개변수로

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))
        ps = predict
        # ps = F.softmax(predict, dim=1)  # model logit을 확률로
        # ps = auto_pred
        # top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class

        top_class = (ps >= 0.5).float()
        correct += (top_class == target.reshape(top_class.shape)).sum()
        total += target.size(0)


    train_loss = np.mean(train_loss)
    acc = (correct / total) * 100
    return train_loss, acc


def auto_val_epoch2(model, loader, rot_class, device):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    # class 맞춘 개수
    class_correct = torch.zeros(len(list(rot_class.keys())))
    # 전체 data를 class 별 개수로 정리
    class_total = torch.zeros(len(list(rot_class.keys())))

    # confusion matrix 구하기 위해
    confusion_target = []
    confusion_pred = []

    logit_list = []
    target_list = []
    with torch.no_grad():
        for (org_data, mi_data, class_label, target) in tqdm(loader):
            org_data, mi_data, class_label = org_data.to(device), mi_data.to(device), class_label.to(device).long()
            target = target.to(device).unsqueeze(dim=1).to(torch.float32)

            # single efficient
            # predict = model(org_data)
            a_predict, predict = model(org_data, mi_data, device)

            # loss = criterion(a_predict, class_label)
            loss = criterion_bce(predict, target)
            # loss = loss + loss_event

            val_loss.append(loss.detach().cpu().numpy())
            ps = predict
            # ps = F.softmax(predict, dim=1)  # model logit을 확률로
            # ps = predict
            # if predict < 0.5:
            #     ps =
            top_class = (ps >= 0.5).float()
            # top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
            equals = top_class == target.reshape(top_class.shape)

            confusion_target.extend(target.detach().cpu().numpy())
            confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

            ps = ps.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

                logit_list.append(ps[i])
                target_list.append(target[i])
        val_loss = np.mean(val_loss)
        acc = sum(class_correct) / sum(class_total) * 100
    voting_3 = 0
    voting_3_t = 0
    voting_10 = 0
    voting_10_t = 0
    vote_all = 0
    vote_all_t = 0

    pred_all = []
    pred_3 = []
    pred_10 = []
    target_all = []
    target_3 = []
    target_10 = []
    ##############################
    ############################## voting
    ##############################len(ps[0])
    # print(ps)
    # print(target_list)
    # if event == 'bump':
    # print(ps)
    # for i in range(len(ps[0])):
    for i in range(2):
        # 사람 위치 찾기(index)
        people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
        softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
        vote_all += int(np.where(softmax >= 0.5, 1,0)) == i
        vote_all_t += 1

        pred_all.append(softmax)
        target_all.append(i)
        for x in range(0, len(people_index), 3):
            softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])/3
            voting_3 += int(np.where(softmax_3 >= 0.5, 1,0)) == i
            voting_3_t += 1

            pred_3.append(softmax_3)
            target_3.append(i)
        for x in range(0, len(people_index), 10):
            softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])/10
            voting_10 += int(np.where(softmax_10 >= 0.5, 1,0)) == i
            voting_10_t += 1

            pred_10.append(softmax_10)
            target_10.append(i)

    print(voting_3)
    print(voting_3_t)
    print(voting_10)
    print(voting_10_t)
    print(vote_all)
    print(vote_all_t)
    # return val_loss, acc, confusion_pred, confusion_target
    return val_loss, acc, confusion_pred, confusion_target, (voting_3 / voting_3_t) * 100, (
            voting_10 / voting_10_t) * 100, (vote_all / vote_all_t) * 100, np.array(logit_list), np.array(
        target_list), np.array(pred_3), np.array(target_3), np.array(pred_10), np.array(target_10), np.array(
        pred_all), np.array(target_all)


# criterion2 = nn.L1Loss()
# criterion = nn.MSELoss()
# cr = nn.HuberLoss()
# def single_train_epoch(model, loader, optimizer, device):
#     model.train()
#     train_loss = []
#     y_pred = []
#     y = []
#     y_factor = []
#     bar = tqdm(loader)

#     correct = 0
#     total = 0

#     for i, (org_data, target, event) in enumerate(bar):
#         optimizer.zero_grad()
#         org_data, target = org_data.to(device), target.to(device).long()
#         event = event.to(device).unsqueeze(dim=1).to(torch.float32)
#         # original data + motion data를 model에 매개변수로
#         predict,event_pred = model(org_data)

#         loss = criterion(predict, target)
#         loss_event = criterion_bce(event_pred, event)
#         loss = loss + loss_event
#         loss.backward()

#         optimizer.step()

#         loss_np = loss.detach().cpu().numpy()
#         train_loss.append(loss_np)
#         smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
#         bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

#         ps = F.softmax(predict, dim=1)  # model logit을 확률로

#         top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
#         correct += (top_class == target.reshape(top_class.shape)).sum()
#         total += target.size(0)

#     train_loss = np.mean(train_loss)
#     acc = (correct / total) * 100
#     return train_loss, acc


# def single_val_epoch(model, loader, rot_class, device):
#     '''

#     Output:
#     val_loss, acc,
#     auc   : 전체 데이터베이스로 진행한 validation
#     auc_no_ext: 외부 데이터베이스를 제외한 validation
#     '''

#     model.eval()
#     val_loss = []
#     mse_loss = []
#     weight_loss = []
#     # class 맞춘 개수
#     class_correct = torch.zeros(len(list(rot_class.keys())))
#     # 전체 data를 class 별 개수로 정리
#     class_total = torch.zeros(len(list(rot_class.keys())))

#     # confusion matrix 구하기 위해
#     confusion_target = []
#     confusion_pred = []

#     logit_list = []
#     target_list = []
#     with torch.no_grad():
#         for (org_data, target, event) in tqdm(loader):
#             org_data, target = org_data.to(device), target.to(device).long()
#             event = event.to(device).unsqueeze(dim=1).to(torch.float32)
#             predict, event_pred = model(org_data)

#             #single efficient
#             # predict = model(org_data)

#             loss = criterion(predict.to(torch.float32), target)
#             loss_event = criterion_bce(event_pred, event)
#             loss = loss + loss_event

#             val_loss.append(loss.detach().cpu().numpy())

#             ps = F.softmax(predict, dim=1)  # model logit을 확률로
#             top_p, top_class = ps.topk(1, dim=1)  # model이 맞춘 class
#             equals = top_class == target.reshape(top_class.shape)

#             confusion_target.extend(target.detach().cpu().numpy())
#             confusion_pred.extend(np.concatenate(top_class.detach().cpu().numpy()))

#             ps = ps.detach().cpu().numpy()
#             target = target.detach().cpu().numpy()
#             for i in range(len(target)):
#                 label = target[i]
#                 class_correct[label] += equals[i].item()
#                 class_total[label] += 1

#                 logit_list.append(ps[i])
#                 target_list.append(target[i])
#         val_loss = np.mean(val_loss)
#         acc = sum(class_correct) / sum(class_total) * 100

#     voting_3 = 0
#     voting_3_t = 0
#     voting_10 = 0
#     voting_10_t = 0
#     vote_all = 0
#     vote_all_t = 0
#     ##############################
#     ############################## voting
#     ##############################
#     for i in range(len(ps[0])):
#         # 사람 위치 찾기(index)
#         people_index = list(filter(lambda x: target_list[x] == i, range(len(target_list))))
#         softmax = sum([logit_list[x] for x in people_index]) / len(people_index)
#         vote_all += np.argmax(softmax) == i
#         vote_all_t += 1
#         for x in range(0, len(people_index), 3):
#             softmax_3 = sum([logit_list[j] for j in people_index[x:x + 3]])
#             voting_3 += np.argmax(softmax_3) == i
#             voting_3_t += 1
#         for x in range(0, len(people_index), 10):
#             softmax_10 = sum([logit_list[j] for j in people_index[x:x + 10]])
#             voting_10 += np.argmax(softmax_10) == i
#             voting_10_t += 1
#     print(voting_3)
#     print(voting_3_t)
#     print(voting_10)
#     print(voting_10_t)
#     print(vote_all)
#     print(vote_all_t)
#     # return val_loss, acc, confusion_pred, confusion_target
#     return val_loss, acc, confusion_pred, confusion_target, (voting_3/voting_3_t)*100, (voting_10/voting_10_t)*100, (vote_all/vote_all_t)*100, np.array(logit_list), np.array(target_list)
