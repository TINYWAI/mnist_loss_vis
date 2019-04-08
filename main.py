import argparse
import os
import yaml
import shutil
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from lossfunc.center_loss import CenterLoss
from lossfunc.ring_loss import RingLoss
from lossfunc.arcface import ArcfaceLoss
from lossfunc.feature_incay import ReciprocalNormLoss
from lossfunc.asoftmax import AngleLoss
from model import Net
from utils import visualize2, visualize2_3D

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='cfgs/center_loss.yaml')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
    config = EasyDict(config)

torch.manual_seed(config.SEED)
if use_cuda:
    torch.cuda.manual_seed(config.SEED)

if not os.path.exists('experiment'):
    os.makedirs('experiment')

exp_path = os.path.join('experiment', config.EXP_NAME)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
shutil.copy(args.config, exp_path) # copy config file

acc_file = os.path.join(exp_path, 'acc.csv') # add acc.csv
with open(acc_file, 'a') as fp:
    fp.write('Epoch, train_acc, test_acc\r\n')

imgs_path = os.path.join(exp_path, 'images')
if not os.path.exists(imgs_path):
    os.makedirs(imgs_path)

def main():
    # Dataset
    trainset = datasets.MNIST('../MNIST', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    testset = datasets.MNIST('../MNIST', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)

    if 'VIS' not in config:
        DIM = 2
    elif config.VIS == '2D':
        DIM = 2
    elif config.VIS == '3D':
        DIM = 3
    # Model
    model = Net(config, feature_dim=DIM)
    model = model.to(device)

    if 'MAIN_LOSS' in config:
        if config.MAIN_LOSS.TYPE == 'arcface':
            criterion = ArcfaceLoss(s=config.S)
        elif config.MAIN_LOSS.TYPE == 'a-softmax':
            criterion = AngleLoss(base=config.MAIN_LOSS.BASE, gamma=config.MAIN_LOSS.GAMMA, power=config.MAIN_LOSS.POWER,
                                  lambda_min=config.MAIN_LOSS.LAMBDA_MIN, latest_iter=0)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)


    optimizer = optim.SGD(model.parameters(), lr=config.BASE_LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    sheduler = lr_scheduler.StepLR(optimizer, config.LR_UPDATE, gamma=config.LR_MULTS)

    if 'ADDITIONAL_LOSS' in config:
        global optimzer4center
        if config.ADDITIONAL_LOSS.TYPE == 'center_loss':
            global centerloss
            centerloss = CenterLoss(num_classes=10, feat_dim=DIM,
                                    sphere_type=config.ADDITIONAL_LOSS.SPHERE.TYPE, R=config.ADDITIONAL_LOSS.SPHERE.R,
                                    size_average=True, metric_mode=config.ADDITIONAL_LOSS.METRIC_MODE,
                                    force=config.ADDITIONAL_LOSS.FORCE).to(device)
            optimzer4center = optim.SGD(centerloss.parameters(), lr=config.ADDITIONAL_LOSS.CENTER_LR)
        elif config.ADDITIONAL_LOSS.TYPE == 'ring_loss':
            global ringloss
            ringloss = RingLoss().to(device)
            optimzer4center = optim.SGD(ringloss.parameters(), lr=config.ADDITIONAL_LOSS.R_LR)
        elif config.ADDITIONAL_LOSS.TYPE == 'reciprocal_norm_loss':
            global reciprocal_norm_loss
            reciprocal_norm_loss = ReciprocalNormLoss().to(device)


    for epoch in range(100):
        sheduler.step()
        # print optimizer4nn.param_groups[0]['lr']

        train_feat, train_label, train_acc = train(epoch + 1, model, train_loader, criterion, optimizer)
        print('train_acc: %.3f%%'%train_acc)
        test_feat, test_label, test_acc = m_test(epoch + 1, model, test_loader)
        print('test_acc: %.3f%%'%test_acc)
        print()
        with open(acc_file, 'a') as fp:
            fp.write('%d, %.3f%%, %.3f%%\r\n'%(epoch, train_acc, test_acc))

        if 'VIS' not in config:
            visualize2(train_feat, train_label, test_feat, test_label, epoch, imgs_path, train_acc, test_acc)
        elif config.VIS == '2D':
            visualize2(train_feat, train_label, test_feat, test_label, epoch, imgs_path, train_acc, test_acc)
        elif config.VIS == '3D':
            visualize2_3D(train_feat, train_label, test_feat, test_label, epoch, imgs_path, train_acc, test_acc)


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    correct = 0
    print("Training... Epoch = %d" % epoch)
    ip1_loader = []
    idx_loader = []
    for i,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ip1, pred = model(data, target)
        if 'ADDITIONAL_LOSS' in config:
            if config.ADDITIONAL_LOSS.TYPE == 'center_loss':
                ct_loss = centerloss(ip1, target)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * ct_loss
            elif config.ADDITIONAL_LOSS.TYPE == 'ring_loss':
                rg_loss = ringloss(ip1)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * rg_loss
            # elif config.MAIN_LOSS.TYPE == 'regularface':
            #     logits, Sep = pred
            #     loss = criterion(logits, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * Sep
            elif config.ADDITIONAL_LOSS.TYPE == 'reciprocal_norm_loss':
                rn_loss = reciprocal_norm_loss(ip1)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * rn_loss
            elif config.ADDITIONAL_LOSS.TYPE == 'fixed_ring_loss':
                frg_loss = fixed_ring_loss(ip1)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * frg_loss
            elif config.ADDITIONAL_LOSS.TYPE == 'fixed_out_ring_loss':
                forg_loss = fixed_out_ring_loss(ip1)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * forg_loss
            elif config.ADDITIONAL_LOSS.TYPE == 'out_ring_loss':
                org_loss = out_ring_loss(ip1)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * org_loss
            elif config.ADDITIONAL_LOSS.TYPE == 'sout_ring_loss':
                sorg_loss = sout_ring_loss(ip1, target)
                loss = criterion(pred, target) + config.ADDITIONAL_LOSS.LOSS_WEIGHT * sorg_loss
        else:
            loss = criterion(pred, target)

        if 'MAIN_LOSS' in config:
            if config.MAIN_LOSS.TYPE == 'arcface':
                result = pred[0].data.max(1, keepdim=True)[1]
            elif config.MAIN_LOSS.TYPE == 'a-softmax':
                result = pred[0].data.max(1, keepdim=True)[1]
            else:
                result = pred.data.max(1, keepdim=True)[1]
        else:
            result = pred.data.max(1, keepdim=True)[1]
        correct += result.eq(target.data.view_as(result)).float().cpu().sum().numpy()

        if i%100 == 0:
            if 'ADDITIONAL_LOSS' in config:
                if config.ADDITIONAL_LOSS.TYPE == 'center_loss':
                    print('loss: %f, ct_loss: %f'%(loss, ct_loss))
                # elif config.LOSS_TYPE == 'regularface':
                #     print('loss: %f, Sep: %f'%(loss, Sep))
                elif config.ADDITIONAL_LOSS.TYPE == 'ring_loss':
                    print('loss: %f, rg_loss: %f'%(loss, rg_loss))
                elif config.ADDITIONAL_LOSS.TYPE == 'reciprocal_norm_loss':
                    print('loss: %f, rn_loss: %f'%(loss, rn_loss))
                elif config.ADDITIONAL_LOSS.TYPE == 'fixed_ring_loss':
                    print('loss: %f, frg_loss: %f'%(loss, frg_loss))
                elif config.ADDITIONAL_LOSS.TYPE == 'fixed_out_ring_loss':
                    print('loss: %f, forg_loss: %f'%(loss, forg_loss))
                elif config.ADDITIONAL_LOSS.TYPE == 'out_ring_loss':
                    print('loss: %f, org_loss: %f'%(loss, org_loss))
                elif config.ADDITIONAL_LOSS.TYPE == 'sout_ring_loss':
                    print('loss: %f, sorg_loss: %f'%(loss, sorg_loss))
            else:
                print('loss: %f'%(loss))

        optimizer.zero_grad()
        if 'ADDITIONAL_LOSS' in config:
            if config.ADDITIONAL_LOSS.TYPE == 'center_loss' or config.ADDITIONAL_LOSS.TYPE == 'ring_loss':
                optimzer4center.zero_grad()

        loss.backward()

        optimizer.step()
        if 'ADDITIONAL_LOSS' in config:
            if config.ADDITIONAL_LOSS.TYPE == 'center_loss' or config.ADDITIONAL_LOSS.TYPE == 'ring_loss':
                optimzer4center.step()

        ip1_loader.append(ip1)
        # ip1_loader.append(pred)
        idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    train_acc = 100. * correct/len(train_loader.dataset)
    # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)
    return feat.data.cpu().numpy(), labels.data.cpu().numpy(), train_acc

def m_test(epoch, model, test_loader):
    model.eval()
    correct = 0
    print("Testing... Epoch = %d" % epoch)
    with torch.no_grad():
        ip1_loader = []
        idx_loader = []
        for i,(data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            ip1, pred = model(data, target)

            if 'MAIN_LOSS' in config:
                if config.MAIN_LOSS.TYPE == 'arcface':
                    result = pred[0].data.max(1, keepdim=True)[1]
                elif config.MAIN_LOSS.TYPE == 'a-softmax':
                    result = pred[0].data.max(1, keepdim=True)[1]
                else:
                    result = pred.data.max(1, keepdim=True)[1]
            else:
                result = pred.data.max(1, keepdim=True)[1]
            correct += result.eq(target.data.view_as(result)).float().cpu().sum().numpy()

            ip1_loader.append(ip1)
            idx_loader.append((target))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    test_acc = 100. * correct/len(test_loader.dataset)

    return feat.data.cpu().numpy(), labels.data.cpu().numpy(), test_acc


if __name__ == '__main__':
    main()