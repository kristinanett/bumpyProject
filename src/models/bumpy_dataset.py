import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import rosbag
import yaml
from cv_bridge import CvBridge
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd


def get_sample_count_by_file(filepath):
    input_bag = rosbag.Bag(filepath, "r") 
    info_dict = yaml.load(input_bag._get_yaml_info(), Loader=yaml.FullLoader)
    return info_dict["messages"]

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

        #normalizing the data
        image = (image-(255/2))/(255/2)
        lin_coms = (lin_coms-0.5)/0.05 #assuming lin coms in range 0.45 to 0.55
        ang_coms = (ang_coms/0.4) #assuming ang coms in range -0.4 to 0.4
        
        return {'image': image,
                'lin_coms': lin_coms,
                'ang_coms': ang_coms,
                'imu_data': imu_data}


class BumpyDataset(Dataset):
    def __init__(self, csv_file, bagfile_dir, transform=None):

        """
        Args:
            csv_file (string): Path to the csv file with imu and command data
            bagfile_dir (string): Directory with all the processed bagfiles with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_df = pd.read_csv(csv_file, header=0)
        self.dir = bagfile_dir
        file_list = glob.glob(self.dir + "/*.bag") #list of filenames ending with .bag
        self.files = sorted((f, get_sample_count_by_file(f)) for f in file_list) #list of tuples with file name and nr of imgs
        self._sample_count = sum(f[-1] for f in self.files) #overall nr of images
        self.transform = transform

    def __len__(self):
        return self._sample_count

    def __getStats__(self):

        return self._sample_count

    def __getitem__(self, idx):
        bridge = CvBridge()
        current_count = 0
        for file_, sample_count in self.files:
            if current_count <= idx < current_count + sample_count:
                # stop when the index we want is in the range of the sample in this file
                break  # now file_ will be the file we want
            current_count += sample_count

        # now file_ has sample_count samples
        file_idx = idx - current_count  # the index we want to access in file_
        with rosbag.Bag(file_, 'r') as f:
            i=0
            for topic, msg, t in f.read_messages(topics=["/cam1/image_rect/compressed"]):
                if i == file_idx:
                    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images
                    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    break
                i=i+1
        
        lin_coms = np.array([self.csv_df.iloc[idx, :8]])
        ang_coms = np.array([self.csv_df.iloc[idx, 8:16]])
        imu_data = np.array([self.csv_df.iloc[idx, 16:]])
        sample = {'image': rgb_img, 'lin_coms': lin_coms, 'ang_coms': ang_coms, 'imu_data': imu_data}

        if self.transform:
            sample = self.transform(sample)

        coms = torch.transpose(torch.cat((sample['lin_coms'], sample['ang_coms']), 0), 0, 1)

        return [sample['image'], coms], sample['imu_data']

#Some code to test the dataset is working properly
# dataset1 = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([Normalize(), ToTensor()]))
# dataloader1 = DataLoader(dataset1, batch_size=1)
# dataloader_iter1 = iter(dataloader1)

# dataset2 = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([ToTensor()]))
# dataloader2 = DataLoader(dataset2, batch_size=1)
# dataloader_iter2 = iter(dataloader2)

# for i in range(1):
#     x1, y1 = next(dataloader_iter1)
#     x2, y2 = next(dataloader_iter2)

# print(x2[0].type())
# print(x2[1].type())
# print(y2)

# print(x1[0].type())
# print(x1[1].type())
# print(y1)

#print(x1[0].size()) #([1, 3, 732, 1490])
#print(x1[1].size()) #[1, 8, 2])
#print(y1.size()) #([1, 1, 8])

# #visualizing the image
# img_to_show = np.moveaxis(img[-1].numpy(), 0, -1) #(732, 1490, 3)
# plt.imshow(img_to_show, cmap="gray")
# plt.show()