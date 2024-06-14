import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

# 追加部分(T.Lambdaがマルチプロセッシングでプロセス間で渡される際にpickleでシリアライズできないため、通常の関数定義に置き換える。)
# TenCropを適用した後にToTensorを適用する関数
def apply_totensor(crops):
    return [T.ToTensor()(crop) for crop in crops]

# TenCropを適用した後にNormalizeを適用する関数
def apply_normalize(crops):
    mean_pix = [x/255.0 for x in [111.69354786878293, 106.14439030589492, 85.24301897108808]]
    std_pix = [x/255.0 for x in [43.723557928469745, 38.893488288905196, 40.978335540439055]]
    return torch.stack([T.Normalize(mean=mean_pix, std=std_pix)(crop) for crop in crops])

class colorWCS(Dataset):
    """

    re-implement to load data from Spyros Gidaris and Niko Komodakis

"""
    def __init__(self, root, resize, split, mode, augment, img_size=224):

        self.resize = resize
        self.split = split
        self.mode = mode

        mean_pix = [x/255.0 for x in [111.69354786878293, 106.14439030589492, 85.24301897108808]]
        std_pix = [x/255.0 for x in [43.723557928469745, 38.893488288905196, 40.978335540439055]]

        padding = 20 # default 8 , im_size = 84
        print("augment in dataloader:",augment)
        if augment == 0:
            self.transform = T.Compose([
                # T.Resize((img_size, img_size)),
                T.Resize((img_size + padding, img_size + padding)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=mean_pix, std=std_pix)
            ])
        elif augment == 1:
            self.transform = T.Compose([
                T.Resize((img_size+padding, img_size+padding)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                T.ToTensor(),
                T.Normalize(mean=mean_pix, std=std_pix)
            ])
        # ten cropを使って10枚に増やしている。
        elif augment == 2:
            self.transform = T.Compose([
                T.Resize((img_size + padding, img_size + padding)),
                T.TenCrop(img_size),
                T.Lambda(apply_totensor),
                T.Lambda(apply_normalize)
                # T.Lambda(lambda crops: [T.ToTensor()(crop) for crop in crops]),
                # T.Lambda(lambda crops: torch.stack([T.Normalize(mean=mean_pix, std=std_pix)(crop) for crop in crops]))
            ])
        else:
            raise NameError('Augment mode {} not implemented.'.format(augment))

        # ここのパスを変える。
        # self.path = 'D:/GraduationProject/Dataset/WCS_Camera_Traps/dataset/infrared_dataset/
        wcs_path = 'E:/IFOR/model/ranma/dataset/wcs/dataset/dataset/color_dataset/'
        cct_path = "E:/IFOR/model/ranma/dataset/cct/dataset/dataset/color_dataset/"
        if self.mode == 'openfew':
            if self.split == 'train':
                file_name = os.path.join(wcs_path, 'train')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            elif self.split == 'val':
                # file_name = os.path.join(root, 'train')
                # self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(cct_path, 'test')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
            else:  # self.split == 'test'
                # file_name = os.path.join(root, 'train')
                # self.data = datasets.ImageFolder(file_name, transform=self.transform)
                file_name = os.path.join(cct_path, 'test')
                self.data = datasets.ImageFolder(file_name, transform=self.transform)
        else:
            raise NameError('Unknown mode ({})!'.format(self.mode))
        self.classes = self.data.classes
        self.cls_num = len(self.classes)
        self.closed_samples = len(self.data)
        self.open_classes = []
        self.open_samples = 0
        self.open_cls_num = len(self.open_classes)
        # if (self.mode == 'openfew' or self.mode == 'openmany') and (self.split == 'test' or self.split == 'val'):
            # self.open_samples = len(self.open_data)
            # self.open_classes = self.open_data.classes
            # self.open_cls_num = len(self.open_classes)

        # train_val_sample_list = torch.zeros(64).long()
        # if self.split == 'val':
        #     for i in range(len(self.data)):
        #         train_val_sample_list[self.data[i][1]] += 1
        # torch.save(train_val_sample_list, 'train_val_sample_list.pt')
        #
        # train_train_sample_list = 600 * torch.ones(64).long()
        # train_val_sample_list = 300 * torch.ones(64).long()
        # train_test_sample_list = 300 * torch.ones(64).long()
        # val_sample_list = 600 * torch.ones(16).long()
        # test_sample_list = 600 * torch.ones(20).long()

        train_train_sample_list = torch.load('E:/IFOR/model/ranma/dataset/wcs/dataset/color_train_sample_list.pt')
        # train_val_sample_list = torch.load('D:\GraduationProject\Dataset\WCS_Camera_Traps\dataset\color_train_sample_list.pt')
        # train_test_sample_list = torch.load('D:\GraduationProject\Dataset\WCS_Camera_Traps\dataset\color_train_sample_list.pt')
        val_sample_list = torch.load('E:/IFOR/model/ranma/dataset/wcs/dataset/color_test_sample_list.pt')
        test_sample_list = torch.load('E:/IFOR/model/ranma/dataset/wcs/dataset/color_test_sample_list.pt')


        if self.split == 'train':
            self.n_sample_list = train_train_sample_list
        elif self.split == 'val':
            if self.mode.startswith('open'):
                self.n_sample_list = val_sample_list
            else:
                self.n_sample_list = val_sample_list
        else:  # self.split == 'test'
            if self.mode.startswith('open'):
                self.n_sample_list = test_sample_list
            else:
                self.n_sample_list = test_sample_list
        # print("n_sample_list",self.n_sample_list)
    def __getitem__(self, index):
        if self.mode == 'regular':
            return self.data[index]
        elif self.mode.startswith('open'):
            # indexが閉じたサンプルの数よりも小さい場合、普通のデータを返す。
            if index < self.closed_samples:
                sample = self.data[index]
                # sample[0]は画像データ、sample[1]はクラスラベル（整数インデックス）
                return sample[0], sample[1]
            # testかvalの場合？
            else:
                # サンプルを追加のデータフォルダから取得している。
                sample = self.open_data[index - self.closed_samples]
                return sample[0], sample[1]+self.cls_num

    def __len__(self):
        if (self.mode == 'openfew' or self.mode == 'openmany') and (self.split == 'test' or self.split == 'val'):
            # closed_samples = len(self.data)
            # open_data = len(datasets.ImageFolder(file_name, transform=self.transform))
            return self.closed_samples + self.open_samples
        else:
            return self.closed_samples
