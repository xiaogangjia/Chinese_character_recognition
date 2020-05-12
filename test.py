import os
import matplotlib
import csv
matplotlib.rc("savefig", dpi=100)
import cv2
import numpy as np
import pandas as pd

import torch
from dataloader import test_data
from torch.utils.data import DataLoader
from data_produce import data, classes
from res2net_v1b import res2net50_v1b

device = torch.device('cuda:0')

img_dir = 'dataset/test/'
tests = os.listdir(img_dir)

testset = test_data(tests, img_dir)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


net = res2net50_v1b(pretrained=False)
net.load_state_dict(torch.load('output/25_params.pkl'))

net.to(device)
net.eval()

results_csv = csv.writer(open('results/test.csv','w'))
results_csv.writerow(['filename', 'label'])

for i, data in enumerate(testloader):
    images, img_names = data
    images = images.to(device)

    print(i)

    preds = net(images)

    preds = preds.detach().cpu().numpy()
    preds = np.squeeze(preds)

    a = np.argsort(preds)[::-1]
    a = a[0:5]

    results = [classes[a[i]] for i in range(5)]
    results = ''.join(results)

    results = [img_names[0], results]

    results_csv.writerow(results)

print('done')