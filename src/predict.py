import torch.nn as nn
import torch
import argparse
import os
import numpy as np
from weathnet import WeatherNet
# from dataset_utils import DeNoiseDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils import get_file_list
import h5py
import timeit
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        
else:
    DEVICE = torch.device('cpu')

VELODYNE_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/our_data/velodyne/"]
LIVOX_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/our_data/livox/"]


class Predictor(object):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.load_model()
                
        if self.opt.data == 'velodyne16':
            self.file_list = get_file_list(VELODYNE_ROOT_DIR_LIST)
            self.H, self.W = 16, 400
        
        elif self.opt.data == 'livox':
            self.file_list = get_file_list(LIVOX_ROOT_DIR_LIST)
            self.H, self.W = 64, 400
        
        self.do_prediction()
    
    def load_model(self):
        # network
        self.model = WeatherNet()
        checkpoint = torch.load(self.opt.checkpoint)
        self.model.load_state_dict(checkpoint)
        print('Loading saved model !')

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.model.to(DEVICE)
        
    
    def load_data(self, file_name):
        """load one single hdf5 file with point cloud data

        the coordinate system is based on the conventions for land vehicles (DIN ISO 8855)
        (https://en.wikipedia.org/wiki/Axes_conventions)

        each channel contains a matrix with 32x400 values, ordered in layers and columns
        e.g. sensorX_1 contains the x-coordinates in a projected 32x400 view
        """

        with h5py.File(file_name, "r", driver='core') as hdf5:
            # for channel in self.channels:
            # self.sensorX_1 = hdf5.get('sensorX_1')[()]
            # self.sensorY_1 = hdf5.get('sensorY_1')[()]
            # self.sensorZ_1 = hdf5.get('sensorZ_1')[()]
            distance_m_1 = hdf5.get('distance_m_1')[()]
            intensity_1 = hdf5.get('intensity_1')[()]
        
        image = np.concatenate((distance_m_1, intensity_1)).reshape((1, 2, self.H, self.W))
        
        return torch.from_numpy(image).type(torch.torch.FloatTensor)
    
    def write_prediction(self, file_name, labels):
        with h5py.File(file_name, 'a') as hf:
            hf.create_dataset('labels_1', data=labels)

    
    def do_prediction(self):
        self.model.eval()

        with torch.no_grad(): 

            # for file_name, data in zip(self.file_list, self.pre_dataloader):
            for file_name in self.file_list:

                start = timeit.default_timer()

                data = self.load_data(file_name)

                images = data.to(DEVICE, dtype=torch.float)
                # images_batch = images.unsqueeze(0)

                predictions = self.model(images)

                predictions = predictions.argmax(dim=1).squeeze().data.cpu()

                self.write_prediction(file_name, np.array(predictions).reshape((self.H, self.W))+100)

                stop = timeit.default_timer()
                print('single inference: ', stop - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='livox')
    parser.add_argument(
        '--checkpoint', type=str, default='../checkpoints/saved_model.pth', help='checkpoint file')
    

    opt = parser.parse_args()
    Predictor(opt)
