import os
import SimpleITK as sitk
import torch
import random
import numpy as np

class MedData_train(torch.utils.data.Dataset):
    def __init__(self, source_dir, label_dir,crop_size=8):
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.patient_dir = os.listdir(label_dir)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dir)

    def rand_crop(self, image_source,image_label):
        t, _, _ = image_source.shape
        new_t = random.randint(0, t - self.crop_size)
        new_image_source = image_source[new_t:new_t + self.crop_size, :, :]
        new_image_label = image_label[new_t:new_t + self.crop_size, :, :]
        return new_image_source,new_image_label

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
        znorm_source_arrary = self.znorm(source_array)
        crop_source_arrary, crop_label_array = self.rand_crop(znorm_source_arrary,label_array)
        out_source_array = crop_source_arrary[:, np.newaxis]
        out_source_array = torch.FloatTensor(out_source_array)
        out_label_array = crop_label_array[:, np.newaxis]
        out_label_array = torch.FloatTensor(out_label_array)
        return out_source_array,out_label_array



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