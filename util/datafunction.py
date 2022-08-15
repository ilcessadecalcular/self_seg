import os
import SimpleITK as sitk
import torch
import random
import numpy as np

class MedData_train_onlycnn(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)
        # self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        image_source = sitk.ReadImage(source_path)
        source_array = sitk.GetArrayFromImage(image_source)
        image_label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(image_label)
        znorm_source_array = self.znorm(source_array)
        # crop_source_array, crop_label_array = self.rand_crop(znorm_source_arrary,label_array)
        out_source_array = znorm_source_array[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        return out_source_array,out_label_array

class MedData_train_flow(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)
        # self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def znorm(self, image):
        mn = image.mean()
        sd = image.std()
        return (image - mn) / sd

    def __getitem__(self, index):
        patient = self.patient_dir[index]
        source_path = os.path.join(self.source_dir, patient)
        label_path = os.path.join(self.label_dir,patient)
        image_source = sitk.ReadImage(source_path)
        source_array = sitk.GetArrayFromImage(image_source)
        image_label = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(image_label)
        out_source_array = source_array[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        return out_source_array,out_label_array


def z_norm(image):
    mn = image.mean()
    sd = image.std()
    return (image - mn) / sd

def normalize_hu(image):
    # 将输入图像的像素值(200 ~ 600)归一化到0~1之间
    image = image.numpy()
    MIN_BOUND = 200
    MAX_BOUND = 600
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image*255).astype(np.uint8)


if __name__ == "__main__":
    source_dir = 'test/source'
    label_dir = 'test/label'
    dataset_train = MedData_train(source_dir,label_dir)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=2)
    for epoch in range(10):
        for ite, sample in enumerate(train_loader):
            img = sample[0]
            label = sample[1]
            # label = target
            print(epoch, ite)
            print(img.size())
            print(label.size())