import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import yaml
from cv_bridge import CvBridge
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data']

        # swap image color axis because -> numpy image: H x W x C but torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'lin_coms': torch.from_numpy(lin_coms).float(),
                'ang_coms': torch.from_numpy(ang_coms).float(),
                'imu_data': torch.from_numpy(imu_data).float()}

class Normalize(object):
    """Normalize the image in the sample"""

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data']

        #normalizing the image data
        image = (image-(255/2))/(255/2)
        
        return {'image': image, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        return {'image': img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data}


class BumpyDataset2(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):

        """
        Args:
            csv_file (string): Path to the csv file with imu and command data
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_df = pd.read_csv(csv_file, header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.stats = self.__getStats__()


    def __len__(self):
        return len(glob.glob(self.img_dir + "*.png"))
       

    def __getStats__(self):
        imu_all = np.array([self.csv_df.iloc[:, 16:]])
        lin_com_all = np.array([self.csv_df.iloc[:, :8]])
        ang_com_all = np.array([self.csv_df.iloc[:, 8:16]])

        imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
        lin_com_mean, lin_com_std = np.mean(lin_com_all), np.std(lin_com_all)
        ang_com_mean, ang_com_std = np.mean(ang_com_all), np.std(ang_com_all)

        return lin_com_mean, lin_com_std, ang_com_mean, ang_com_std, imu_mean, imu_std

    def __getitem__(self, idx):

        #getting the image
        img_name = "frame%06i.png" % idx
        cv_img = cv2.imread(self.img_dir + "/" + img_name)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        #getting the other data and normalizing it right away by subtracting mean and dividing by std found on the whole dataset
        lin_coms = (np.array([self.csv_df.iloc[idx, :8]])-self.stats[0])/self.stats[1]
        ang_coms = (np.array([self.csv_df.iloc[idx, 8:16]])-self.stats[2])/self.stats[3]
        imu_data = (np.array([self.csv_df.iloc[idx, 16:]])-self.stats[4])/self.stats[5]
        sample = {'image': rgb_img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data}

        if self.transform:
            sample = self.transform(sample)

        #some data reshaping
        coms_final = torch.transpose(torch.cat((sample['lin_coms'], sample['ang_coms']), 0), 0, 1)
        imu_final = torch.transpose(sample['imu_data'], 0, 1)

        return [sample['image'], coms_final], imu_final

#Some code to test the dataset is working properly
# dataset1 = BumpyDataset2("data/processed/data3.csv","data/processed/imgs/", transform=transforms.Compose([Normalize(), ToTensor()]))
# dataloader1 = DataLoader(dataset1, batch_size=1)
# dataloader_iter1 = iter(dataloader1)

#dataset2 = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([ToTensor()]))
#dataset2.__getitem__(7360)
#dataloader2 = DataLoader(dataset2, batch_size=32)
#dataloader_iter2 = iter(dataloader2)

# for i in range(1):
#     x1, y1 = next(dataloader_iter1)
#   x2, y2 = next(dataloader_iter2)

# print(x2[0].type())
# print(x2[1].type())
# print(y2)

# print(x1[0].type())
# print(x1[1].type())
# print(y1)

# print(x1[0].size()) #([1, 3, 732, 1490])
# print(x1[1].size()) #[1, 8, 2])
# print(y1.size()) #([1, 8, 1])

#visualizing the image
# img_to_show = np.moveaxis(x2[0][0].numpy(), 0, -1) #(732, 1490, 3)
# plt.imshow(img_to_show, cmap="gray")
# plt.show()