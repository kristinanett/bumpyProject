import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data, cur_imu = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data'], sample['cur_imu']

        # swap image color axis because -> numpy image: H x W x C but torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'lin_coms': torch.from_numpy(lin_coms).float(),
                'ang_coms': torch.from_numpy(ang_coms).float(),
                'imu_data': torch.from_numpy(imu_data).float(),
                'cur_imu': torch.from_numpy(cur_imu).float()}

class NormalizeIMG(object):
    """Normalize the image in the sample"""

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data, cur_imu = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data'], sample['cur_imu']

        #normalizing the image data
        #image = (image-(255/2))/(255/2)
        channel_means = np.mean(image, axis=(0,1))
        channel_stds = np.std(image, axis=(0,1))

        standardized_images_out = (image - channel_means) / channel_stds
        
        return {'image': standardized_images_out, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data, 'cur_imu': cur_imu}


class Crop(object):
    """Crop the upper x pixels of the image out

    Args:
        crop_ratio (float): Crop ratio. How much of the height to crop out from the top (eg 0.5 crops top half of the image)
    """

    def __init__(self, crop_ratio):
        assert isinstance(crop_ratio, (float))
        self.crop_ratio = crop_ratio

    def __call__(self, sample):
        image, lin_coms, ang_coms, imu_data, cur_imu = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data'], sample['cur_imu']

        h, w = image.shape[:2]
        crop_img = image[int(self.crop_ratio*h):h, :].copy()

        return {'image': crop_img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data, 'cur_imu': cur_imu}

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
        image, lin_coms, ang_coms, imu_data, cur_imu = sample['image'], sample['lin_coms'], sample['ang_coms'], sample['imu_data'], sample['cur_imu']

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

        return {'image': img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data, 'cur_imu': cur_imu}


class BumpyDataset(Dataset):
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
        imu_all = np.array([self.csv_df.iloc[:, 17:]])
        lin_com_all = np.array([self.csv_df.iloc[:, 1:9]])
        ang_com_all = np.array([self.csv_df.iloc[:, 9:17]])

        imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
        lin_com_mean, lin_com_std = np.mean(lin_com_all), np.std(lin_com_all)
        ang_com_mean, ang_com_std = np.mean(ang_com_all), np.std(ang_com_all)
        #print("Stats:", lin_com_mean, lin_com_std, ang_com_mean, ang_com_std, imu_mean, imu_std)

        return lin_com_mean, lin_com_std, ang_com_mean, ang_com_std, imu_mean, imu_std

    def __getitem__(self, idx):

        #getting the image
        img_name = "frame%06i.png" % idx
        cv_img = cv2.imread(self.img_dir + "/" + img_name)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        #getting the other data and normalizing it right away by subtracting mean and dividing by std found on the whole dataset
        lin_coms = (np.array([self.csv_df.iloc[idx, 1:9]])-self.stats[0])/self.stats[1]
        ang_coms = (np.array([self.csv_df.iloc[idx, 9:17]])-self.stats[2])/self.stats[3]
        imu_data = (np.array([self.csv_df.iloc[idx, 17:]])-self.stats[4])/self.stats[5]
        cur_imu = (np.array([self.csv_df.iloc[idx, 0]])-self.stats[4])/self.stats[5]
        sample = {'image': rgb_img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data, 'cur_imu': cur_imu}

        if self.transform:
            sample = self.transform(sample)

        #some data reshaping
        coms_final = torch.transpose(torch.cat((sample['lin_coms'], sample['ang_coms']), 0), 0, 1)
        imu_final = torch.transpose(sample['imu_data'], 0, 1)

        return [sample['image'], coms_final, sample['cur_imu']], imu_final, idx

#Some code to test the dataset is working properly
# dataset1 = BumpyDataset("/work3/s203129/0405and1605/data.csv","/work3/s203129/0405and1605/imgs/", transform=transforms.Compose([Rescale(366), Crop(0.45), NormalizeIMG(), ToTensor()]))
# dataloader1 = DataLoader(dataset1, batch_size=1)
# dataloader_iter1 = iter(dataloader1)

# dataset2 = BumpyDataset("data/processed/data.csv","data/processed/imgs/", transform=transforms.Compose([Rescale(122), Crop(0.45), NormalizeIMG(), ToTensor()])) #68x248aftercrop
# #dataset2.__getitem__(7360)
# dataloader2 = DataLoader(dataset2, batch_size=1)
# dataloader_iter2 = iter(dataloader2)

# for i in range(1):
    #inputs, labels, idx = next(dataloader_iter1)
    # inputs, labels, idx = next(dataloader_iter2)

# print(x2[0].type())
# print(x2[1].type())
# print(y2)

#print(x1[0].type())
#print(x1[1].type())
# print(x1[0])
# print(x1[1])
# print(y1)

# print(inputs[0].size()) #image: ([1, 3, 732, 1490])
# print(inputs[1].size()) #commands: [1, 8, 2])
# print(inputs[2].size()) #current_imu: ([1, 1])
# print(labels.size()) #target imu values: ([1, 8, 1])

# img_batch = x2[0].numpy()
# print(np.shape(img_batch))
# means = np.mean(img_batch, axis=(0,2,3))
# print(np.shape(means))
# print(means)

# print(x2[0].size()) #torch.Size([1, 3, 68, 248]
# print(x2[0][0].size()) #torch.Size([3, 68, 248])
# print(x2[0][0].numpy())

# #visualizing the image
# img_to_show = np.moveaxis(x2[0][0].numpy(), 0, -1).astype(int) #(732, 1490, 3)
# plt.imshow(img_to_show, cmap="gray")
# plt.show()