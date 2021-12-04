import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
from utils import get_file_list
# TRAIN_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/train_01/",
#                        "/home/newdisk/yanqiao/dataset/cnn_denoising/train_02/",
#                        "/home/newdisk/yanqiao/dataset/cnn_denoising/train_road_01/",
#                        "/home/newdisk/yanqiao/dataset/cnn_denoising/train_road_02/"]
# TEST_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/test_01/"]
# VAL_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/val_01/"]


TRAIN_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_02/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_01/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_02/"]
TEST_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/test_01/"]
VAL_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/val_01/"]



def normalize(x):
    return ( 255 * (x-np.min(x)) / (np.max(x)-np.min(x)) ).astype(np.uint8)

class DeNoiseDataset(Dataset):
    def __init__(self, mode):

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


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        distance_m_1, intensity_1, labels_1 = self.load_hdf5_file(file_path)
        image =  np.concatenate((distance_m_1, intensity_1)).reshape((2,32,400))
        label = labels_1.astype(np.uint8)

        return torch.from_numpy(image).type(torch.torch.FloatTensor), torch.from_numpy(label).type(torch.torch.FloatTensor)
    
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
        return distance_m_1, intensity_1, process_label(labels_1)

def process_label(labels):
    process_labels = []
    for label in labels.flatten():
        if label == 0 or label == 100:
            process_labels.append(0)
        elif label == 101:
            process_labels.append(1)
        else:
            process_labels.append(2)
    return np.array(process_labels).reshape((32, 400))