import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from dataset.image import random_crop, _resize_image, _clip_detections, color_jittering_, lighting_, normalize_
from data_produce import classes


class VOC_data(Dataset):

    def __init__(self, img_dict, img_size=(224, 224), num_class=100):
        self.img_label = img_dict
        self.img_names = list(img_dict.keys())

        self.img_size = img_size
        self.num_class = num_class

        self._data_rng = np.random.RandomState(123)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self.eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
    # return image and its label

    def __getitem__(self, index):
        img_name = self.img_names[index]

        cls = self.img_label[img_name]

        cls_name = classes[cls]
        image = cv2.imread('dataset/train/' + cls_name + '/' + img_name + '.jpg')

        image = cv2.resize(image, self.img_size)

        image = image.astype(np.float32) / 255.
        color_jittering_(self._data_rng, image)
        lighting_(self._data_rng, image, 0.1, self.eig_val, self.eig_vec)
        normalize_(image, self.mean, self.std)
        image = image.transpose((2, 0, 1))

        labels = np.zeros(self.num_class, dtype=np.float32)
        labels[cls] = 1

        image = torch.from_numpy(image)
        labels = torch.from_numpy(labels)

        return image, labels, img_name

    def __len__(self):
        return len(self.img_names)


class test_data(Dataset):
    def __init__(self, img_list, data_dir, img_size=(224, 224)):
        self.img_list = img_list
        self.img_dir = data_dir
        self.img_size = img_size

        self._data_rng = np.random.RandomState(123)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self.eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        image = cv2.imread(self.img_dir + img_name)

        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.
        #color_jittering_(self._data_rng, image)
        #lighting_(self._data_rng, image, 0.1, self.eig_val, self.eig_vec)
        normalize_(image, self.mean, self.std)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)

        return image, img_name

    def __len__(self):
        return len(self.img_list)
