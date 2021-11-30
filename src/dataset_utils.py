import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os

TRAIN_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_02/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_01/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_02/"]
TEST_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/test_01/"]
VAL_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/val_01/"]


def get_file_list(root_dir_list):
    file_list = []
    for root_dir in root_dir_list:
        for dir, root, files in os.walk(root_dir):
            for file in files:
                file_list.append(dir+'/'+file)
    return file_list

def normalize(x):
    return ( 255 * (x-np.min(x)) / (np.max(x)-np.min(x)) ).astype(np.uint8)

class DeNoiseDataset(Dataset):
    def __init__(self, mode, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if mode == 'train':
            self.file_list = get_file_list(TRAIN_ROOT_DIR_LIST)
        elif mode == 'test':
            self.file_list = get_file_list(TEST_ROOT_DIR_LIST)
        else:
            self.file_list = get_file_list(VAL_ROOT_DIR_LIST)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        distance_m_1, intensity_1, labels_1 = self.load_hdf5_file(file_path)
        image =  np.concatenate((distance_m_1, intensity_1)).reshape((2,32,400))
        label = labels_1.astype(np.uint8)

        # if self.transform:
        #     sample = self.transform(sample)

        return torch.from_numpy(image), torch.from_numpy(label)
    
    def load_hdf5_file(self, filename):
        with h5py.File(filename, "r", driver='core') as hdf5:
            # for channel in self.channels:
            # sensorX_1 = hdf5.get('sensorX_1')[()]
            # sensorY_1 = hdf5.get('sensorY_1')[()]
            # sensorZ_1 = hdf5.get('sensorZ_1')[()]
            distance_m_1 = hdf5.get('distance_m_1')[()]
            intensity_1 = hdf5.get('intensity_1')[()]
            labels_1 = hdf5.get('labels_1')[()]

        # sensorX_1 = normalize(sensorX_1)
        # sensorY_1 = normalize(sensorY_1)
        # sensorZ_1 = normalize(sensorZ_1)
        distance_m_1 = normalize(distance_m_1)
        intensity_1 = normalize(intensity_1)
        labels_1 -= 99


        return distance_m_1, intensity_1, labels_1

