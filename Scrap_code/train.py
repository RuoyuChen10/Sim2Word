# -*- coding: utf-8 -*-  

"""
Created on 2021/1/31

@author: Ruoyu Chen
"""

import argparse
import os

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Datasets.dataload as dl
from utils import *

from tqdm import tqdm

class MultiClassLoss(nn.Module):
    def __init__(self):
        super(MultiClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, outs, labels):
        loss = 0
        loss_information = []
        for out,label in zip(outs,labels):
            criterion_loss = self.criterion(out, label)
            loss += criterion_loss
            loss_information.append(criterion_loss.data.item())
        return loss,loss_information

def Compute_Accuracy(out1,out2,out3,label1,label2,label3):
    '''
    Compute the accuracy
        out: output
        label: label
    '''
    pred1 = out1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred2 = out2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred3 = out3.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct1 = pred1.eq(label1.view_as(pred1)).sum().item()/len(out1)
    correct2 = pred2.eq(label2.view_as(pred2)).sum().item()/len(out2)
    correct3 = pred3.eq(label3.view_as(pred3)).sum().item()/len(out3)
    return correct1*100,correct2*100,correct3*100


def optimize_param(model, train_loader, optimizer, loss, datasets_path, epoch):
    '''
    Optimize the parameters
        model: the model
        train_loader: dataloader include the txt information
        optimizer: optimization method
        loss: Loss function
        datasets-path: Path to the datasets
        epoch: the epoch of training
    '''
    model.train()
    train_step = tqdm(train_loader)
    for data in train_step:
        try:
            # Load the data
            train_data, label1, label2, label3 = dl.analysis_data(data,datasets_path)
            # GPU
            if torch.cuda.is_available():
                train_data = torch.cuda.FloatTensor(train_data)
                label1 = torch.cuda.LongTensor(label1)
                label2 = torch.cuda.LongTensor(label2)
                label3 = torch.cuda.LongTensor(label3)
            else:
                train_data = Variable(torch.FloatTensor(train_data))
                label1 = Variable(torch.LongTensor(label1))
                label2 = Variable(torch.LongTensor(label2))
                label3 = Variable(torch.LongTensor(label3))
            # Output
            out1,out2,out3 = model(train_data)
            # Loss
            losses,loss_information = loss([out1,out2,out3],[label1,label2,label3])
            # Accuracy
            correct1,correct2,correct3 = Compute_Accuracy(out1,out2,out3,label1,label2,label3)
            # Information
            train_step.set_description("Epoch %d training set: Total loss: %f, loss1: %f, loss2: %f, loss3: %f, acc1: %f%%, acc2: %f%%, acc3: %f%%." \
                % (epoch,losses.data.item(),loss_information[0],loss_information[1],loss_information[2],correct1,correct2,correct3))
            # Optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # Empty the CUDA menmory
            torch.cuda.empty_cache()
        except RuntimeError as exception:
            # if out of memory
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    torch.save(model, "./checkpoint/model-"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'.pth')
                    raise exception
            else:
                raise exception

def eval_model(model, val_loader, loss, datasets_path, epoch):
    '''
    Evaluate the model
        model: the model
        val_loader: dataloader include the txt information
        loss: Loss function
        datasets-path: Path to the datasets
        epoch: the epoch of training
    '''
    model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    val_step = tqdm(val_loader)
    with torch.no_grad():
      for data in val_step:
            try:
                val_data, label1, label2, label3 = dl.analysis_data(data,datasets_path)
                # GPU
                if torch.cuda.is_available():
                    val_data = torch.cuda.FloatTensor(val_data)
                    label1 = torch.cuda.LongTensor(label1)
                    label2 = torch.cuda.LongTensor(label2)
                    label3 = torch.cuda.LongTensor(label3)
                else:
                    val_data = Variable(torch.FloatTensor(val_data))
                    label1 = Variable(torch.LongTensor(label1))
                    label2 = Variable(torch.LongTensor(label2))
                    label3 = Variable(torch.LongTensor(label3))
                # Output
                out1,out2,out3 = model(val_data)
                # Loss
                pred1 = out1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                pred2 = out2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                pred3 = out3.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct1 += pred1.eq(label1.view_as(pred1)).sum().item()
                correct2 += pred2.eq(label2.view_as(pred2)).sum().item()
                correct3 += pred3.eq(label3.view_as(pred3)).sum().item()
                val_step.set_description(
                    "Epoch %d validation set: acc1: %f%%, acc2: %f%%, acc3: %f%%." \
                    % (epoch, 
                        pred1.eq(label1.view_as(pred1)).sum().item()/len(out1),
                        pred2.eq(label2.view_as(pred2)).sum().item()/len(out2),
                        pred3.eq(label3.view_as(pred3)).sum().item()/len(out3)
                    )
                )
                # Empty the CUDA menmory
                torch.cuda.empty_cache()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.save(model, "./checkpoint/model-"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'.pth')
                        torch.cuda.empty_cache()
                        raise exception
                else:
                    raise exception
    val_step.set_description('Epoch %d validation set: Accuracy1: {}/{} ({:.0f}%), Accuracy2: {}/{} ({:.0f}%), Accuracy3: {}/{} ({:.0f}%)'.format(
        epoch,
        correct1, len(val_loader.dataset), 100. * correct1 / len(val_loader.dataset),
        correct2, len(val_loader.dataset), 100. * correct2 / len(val_loader.dataset),
        correct3, len(val_loader.dataset), 100. * correct3 / len(val_loader.dataset)
        )
    )

def load_pretrained(weight_path):
    model = torch.load(weight_path)
    try:
        # if multi-gpu model:
        model = model.module
    except:
        # just 1 gpu or cpu
        pass
    pretrain = model.state_dict()
    new_state_dict = {}
    for k,v in pretrain.items():
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("Model parameters: " + weight_path + " has been load!")
    return model

def train(args):
    '''
    Train the network
    '''
    # Load the input dir
    train_data_dir = dl.data_dir_read(args.train_path)
    val_data_dir = dl.data_dir_read(args.val_path)
    # DataLoader
    train_loader = DataLoader(train_data_dir, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_dir, batch_size=args.batch_size, shuffle=False)
    
    # Pretrained
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        model = load_pretrained(args.pretrained_path)
    else:
        # Network
        model = get_network(args.network,args.pretrained_path)
            
    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Loss function
    loss = MultiClassLoss()
    # Optimization method
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.01)

    for epoch in range(1, args.epoch+1):
        optimize_param(model, train_loader, optimizer, loss, args.datasets_path, epoch)
        torch.save(model, "./checkpoint/model-item-epoch-"+str(epoch)+'.pth')
        eval_model(model, val_loader, loss, args.datasets_path, epoch)
    torch.save(model, args.save_path)

def main():
    parser = argparse.ArgumentParser("PyTorch Privacy Recognition")
    parser.add_argument('--train-path', type=str, default="./Datasets/training.txt",
                        help='Train path')
    parser.add_argument('--val-path', type=str, default="./Datasets/val.txt",
                        help='Validation path')
    parser.add_argument('--datasets-path', type=str, 
                        default="/home/cry/data1/VGGFace2-pytorch/VGGFace2/train/",
                        help='Path to the datasets')
    parser.add_argument('--pretrained-path', type=str, default="./checkpoint/model.pth",
                        help='Path to the pretrained model.')
    parser.add_argument('--network', type=str,
                        choices=['resnet50'], default="resnet50",
                        help='The network')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--opt', type=str, 
                        choices=['Adam'], default="Adam",
                        help='Optimization method')
    parser.add_argument('--gpu-device', type=str, default="0",
                        help='GPU device')
    parser.add_argument('--save-path', type=str, default="./checkpoint/model.pth",
                        help='Path to save the model.')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    train(args)

if __name__ == '__main__':
    main()
