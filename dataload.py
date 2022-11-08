import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torch.utils
import torchvision

import scipy.io as scio
import torchvision.transforms as transforms
import numpy as np


from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader, mode='train'):
        current_dir = os.getcwd()

        # files = os.listdir('data/KETI/train_val')
        # print(files)
        # json_files = [y for y in files if '.json' in y]

        # fn = scio.loadmat(label)
        '''edited'''
        imgs = []
        if mode == 'train':
            img_files = os.listdir(root)
            img_files = winsort(img_files)
            class_idx = np.loadtxt('classes_ids.csv', dtype=int, delimiter=',')
            class_idx = class_idx[:141096]
            img_files = img_files[:141096]
            anno_file = np.loadtxt("data/KETI/train_all.csv", dtype=int, delimiter=',')

            # testlabel = fn['train_label']
            # testimg = fn['train_images_name']

            testimg = img_files
            testlabel = anno_file
        if mode == 'test':
            # testlabel = fn['test_label']
            # testimg = fn['test_images_name']
            img_files = os.listdir(root)
            img_files = winsort(img_files)
            img_files = img_files[141096:]
            class_idx = np.loadtxt('classes_ids.csv', dtype=int, delimiter=',')
            class_idx = class_idx[141096:]
            anno_file = np.loadtxt("data/KETI/test_all.csv", dtype=int, delimiter=',')

            # testlabel = fn['train_label']
            # testimg = fn['train_images_name']

            testimg = img_files
            testlabel = anno_file
        # if mode == 'validate':
        #     testlabel = fn['val_label']
        #     testimg = fn['val_images_name']
        count = 0
        print(len(testimg))
        for name in testimg:
            # print(name)
            if mode == 'train':
                if os.path.isfile(root + '/' + name):
                    imgs.append((name, testlabel[count], class_idx[count]))
            elif mode == 'test':
                if os.path.isfile(root + '/' + name):
                    imgs.append((name, testlabel[count], class_idx[count]))
            count = count + 1
        self.label = testlabel
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label, class_idx = self.imgs[index]
        img = self.loader(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label), class_idx  # todo: for testing visualization, it needs filename

    def __len__(self):
        return len(self.imgs)


def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.title("bat")
    plt.show()


if __name__ == '__main__':
    mytransform = transforms.Compose([

        transforms.Resize(256),
        transforms.ToTensor(),  # mmb
    ]
    )

    # torch.utils.data.DataLoader
    set = myImageFloder(
        root="data/KETI/train_val",
        label="data/KETI",
        transform=mytransform
    )
    imgLoader = torch.utils.data.DataLoader(
        set,
        batch_size=1, shuffle=False, num_workers=2)

    print(len(set))

    dataiter = iter(imgLoader)
    images, labels = dataiter.next()
    imshow(images)
    print(labels)