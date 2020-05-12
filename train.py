import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from res2net_v1b import res2net50_v1b
from data_produce import data
from dataloader import VOC_data
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda:0')


trains = data()

trainset = VOC_data(trains)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)


if __name__ == '__main__':

    writer = SummaryWriter('runs/res2net')

    dummy_input = torch.rand(1, 3, 224, 224)

    net = res2net50_v1b(pretrained=False)
    net.load_state_dict(torch.load('output/res2net50_v1b_26w_4s-3cf99910.pth'), strict=False)

    print(np.sum([p.numel() for p in net.parameters()]).item())

    writer.add_graph(net, (dummy_input,))

    net.to(device)
    net.train()

    #loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
    #CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    train_num = 0
    test_num = 0
    train_loss = 0
    test_loss = 0

    for epoch in range(0, 30):

        print('epoch: ' + str(epoch))

        if epoch == 20:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1

        for i, data in enumerate(trainloader):
            length = len(trainloader)
            optimizer.zero_grad()

            inputs, labels, img_name= data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = net(inputs)

            #all_loss = loss(preds, labels)
            all_loss = (labels * (-torch.log(preds))).sum() / len(preds)
            all_loss = all_loss.mean()

            all_loss.backward()

            optimizer.step()
            print("training step {}: ".format(i), all_loss.item())

            train_loss += all_loss.item()
            if (i + 1) % 30 == 0:
                writer.add_scalar('training loss',
                                  train_loss / 30,
                                  i + epoch * length)
                train_loss = 0

        if epoch > 0:
            torch.save(net.state_dict(), 'output/{}_params.pkl'.format(epoch))

