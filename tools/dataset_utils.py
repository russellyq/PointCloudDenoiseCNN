#!/usr/bin/python3
from re import S
import numpy as np
import glob
import h5py
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import select
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from utils import normalize
import cv2


PATH = "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/2018-11-29_152024_Static1-Day-FogC/"
COLOR_LABEL_MAPPING = {
    0: [0, 0, 0],       # no lable
    100: [0, 0, 0],     # valid / clear
    101: [255, 0, 0], # rain
    102: [0, 0, 255], # fog
}


class Dataset(object):
    def __init__(self, color_label_mapping=COLOR_LABEL_MAPPING, path=PATH, max_frames=MAX_FRAMES) -> None:
        super().__init__()
        """initialize ros python api with bode 'RosPublisher' and set hdf5 channel names"""
        self.color_label_mapping = color_label_mapping
        self.path = path


        # define hdf5 data format
        self.channels = ['labels_1', 'distance_m_1', 'intensity_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None
        # 读入数据格式 array 32 * 400
        self.load_data()

         
    
    def load_data(self):
        files = sorted(glob.glob(self.path + '*.hdf5'))
        #print('Directory {} contains are {} hdf5-files'.format(self.path, len(files)))

        if len(files) == 0:
            print('Please check the input dir {}. Could not find any hdf5-file'.format(self.path))
        else:
            #print('Start publihsing the first {} frames...'.format(self.max_frames))
            for frame, file in enumerate(files):
                #print('{:04d} / {}'.format(frame, file))

                # load file
                self.load_hdf5_file(file)
                self.publish()

                x_img = normalize(self.sensorX_1)
                y_img = normalize(self.sensorY_1)
                z_img = normalize(self.sensorZ_1)
                range_img = normalize(self.distance_m_1)
                intensity_img = normalize(self.intensity_1)
                





    def load_hdf5_file(self, filename):
        """load one single hdf5 file with point cloud data

        the coordinate system is based on the conventions for land vehicles (DIN ISO 8855)
        (https://en.wikipedia.org/wiki/Axes_conventions)

        each channel contains a matrix with 32x400 values, ordered in layers and columns
        e.g. sensorX_1 contains the x-coordinates in a projected 32x400 view
        """

        with h5py.File(filename, "r", driver='core') as hdf5:
            # for channel in self.channels:
            self.sensorX_1 = hdf5.get('sensorX_1')[()]
            self.sensorY_1 = hdf5.get('sensorY_1')[()]
            self.sensorZ_1 = hdf5.get('sensorZ_1')[()]
            self.distance_m_1 = hdf5.get('distance_m_1')[()]
            self.intensity_1 = hdf5.get('intensity_1')[()]
            self.labels_1 = hdf5.get('labels_1')[()]

