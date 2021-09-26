import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
import json

from tqdm import trange
from easydict import EasyDict
from sklearn.preprocessing import MinMaxScaler

from parse import args
from util import EarlyStopMonitor, write_result, set_random_seed, get_free_gpu, Displayer
from preprocess import loadData

def getModel(args, nfeat_dim=11, efeat_dim=9, gfeat_dim=6):
    num_tasks = 1
    if args.model == "expc":
        from expc.model import Net
        with open('expc/config.json', 'r') as config_file:
            config = EasyDict(json.load(config_file))
        model = Net(config, nfeat_dim, efeat_dim, gfeat_dim, num_tasks).to(args.device)
    elif args.model == 'mlp1':
        from mlp.mlp import MLP1
        with open('mlp/config.json', 'r') as config_file:
            config = EasyDict(json.load(config_file))
        model = MLP1(config, gfeat_dim, num_tasks).to(args.device)
    else:
        from mlp.mlp import MLP
        with open('mlp/config.json', 'r') as config_file:
            config = EasyDict(json.load(config_file))
        model = MLP(config, gfeat_dim, num_tasks).to(args.device)
    return model, config

def evalModel(args, model, loader, criterion):
    model.eval()
    displayer = Displayer(num_data=2, legend=["Predict Force","True Force"], sort_id=1)
    total, total_loss = 0, 0
    for (gfeat, center, diameter, label) in loader:
        gfeat = gfeat.to(args.device)
        center = center.to(args.device)
        diameter = diameter.to(args.device)
        
        outputs = model(gfeat, center, diameter)
        label = label.to(args.device)

        loss = criterion(outputs, label)
        total += len(label)
        total_loss += loss.item() * len(label)

        if not args.using_cpu:
            outs = outputs.cpu().detach().numpy()
            lb = label.cpu().detach().numpy()
        displayer.record([np.squeeze(outs,1), np.squeeze(lb,1)])
    total_loss /= total
    model.train()
    return total_loss, displayer

def trainModel(args, model, trainloader, validloader, criterion, optimizer, scheduler):
    early_stopper = EarlyStopMonitor(max_round=5, higher_better=False)
    loss_displayer = Displayer(num_data=2, legend=["trian loss", "valid loss"], \
            xlabel='Epoch', ylabel='loss')
    outer = trange(args.max_epoch)
    for epoch in outer:
        model.train()
        total, train_loss = 0, 0
        for (gfeat, graph, label) in trainloader:
            gfeat = gfeat.to(args.device)
            print(graph.x.shape)
            print(gfeat)
            graph = graph.to(args.device)
            # center = center.to(args.device)
            # diameter = diameter.to(args.device)
            
            outputs = model(gfeat, center, diameter)
            label = label.to(args.device)

            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += len(label)
            train_loss += loss.item() * len(label)
        train_loss /= total
        valid_loss, _ = evalModel(args, model, validloader, criterion)
        scheduler.step()

        loss_displayer.record([train_loss, valid_loss])
        # if epoch%20 ==0:
        #     print("epoch:{}, trains:{}, valid:{}, lr:{}".format(epoch, train_loss, valid_loss, scheduler.get_last_lr()))
        outer.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
        if epoch>args.min_epoch and early_stopper.early_stop_check(valid_loss):
            break
    return model, train_loss, loss_displayer
    
    
if __name__ == '__main__':
    set_random_seed()
    if args.using_cpu:
        args.device = torch.device('cpu')
    else:
        if args.gpu_id >= 0:
            args.device = torch.device('cuda:{}'.format(args.gpu_id))
        else:
            args.device = torch.device('cuda:{}'.format(get_free_gpu()))

    trainloader, validloader, testloader, label_scaler, gfeat_dim = \
            loadData(args, 0.7, 0.1)
    
    model, config = getModel(args, 11, 6, gfeat_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.2)

    model, train_loss, loss_displayer = trainModel(args, model, trainloader, \
                                validloader, criterion, optimizer, scheduler)
    test_loss, displayer = evalModel(args, model, testloader, criterion)

    if not args.true_label:
        displayer.transform(label_scaler)
    pred_force = torch.tensor(displayer.y[0]).float()
    true_force = torch.tensor(displayer.y[1]).float()
    force_L1loss = nn.L1Loss()(pred_force, true_force)
    print("train_loss:{:.4f}, test_loss:{:.4f},force_L1loss:{:.1f}".format\
        (train_loss, test_loss,force_L1loss))

    if args.draw:
        test_loss, force_L1loss = np.round(test_loss, 4), np.round(force_L1loss, 0)
        save_path = 'jpg/{}_{}_{}_predicts.jpg'.format(args.model, test_loss, force_L1loss)
        title = '{}, {}'.format(test_loss, force_L1loss)
        displayer.plt(show=0, save_path=save_path, title=title)
        save_path = 'jpg/{}_{}_{}_loss.jpg'.format(args.model, test_loss, force_L1loss)
        loss_displayer.plt(mode='plot',show=0, save_path=save_path, title=title)
