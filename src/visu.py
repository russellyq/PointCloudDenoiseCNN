#!/usr/bin/python3
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
from utils import get_file_list, normalize
import cv2
import os

TRAIN_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/2018-11-29_154253_Static1-Day-Rain15",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_02/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_01/",
                       "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_02/"]

TEST_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/test_01/"]

VAL_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/val_01/"]

PREDICTION_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/our_data_predictions/our_data/velodyne/"]

COLOR_LABEL_MAPPING = {
    0: [0, 0, 0],       # no lable
    100: [0, 0, 0],     # valid / clear  -> 0
    101: [255, 0, 0], # rain             -> 1 red
    102: [0, 255, 0], # fog              -> 2 green
}

MAX_FRAMES = 286

class RosPublisher(object):
    def __init__(self, color_label_mapping=COLOR_LABEL_MAPPING, dir_list=TRAIN_ROOT_DIR_LIST, max_frames=MAX_FRAMES) -> None:
        super().__init__()
        """initialize ros python api with bode 'RosPublisher' and set hdf5 channel names"""
        self.color_label_mapping = color_label_mapping
        self.path = get_file_list(dir_list)
        self.max_frames = max_frames

        rospy.init_node('ros_publisher_pointcloud')
        self.ros_rate = rospy.Rate(10)
        self.ros_publisher = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)

        # define hdf5 data format
        self.channels = ['labels_1', 'distance_m_1', 'intensity_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None

        self.idx = 0

        # 读入数据格式 array 32 * 400
        # self.load_data()

        self.publish_data()
    
    def publish_data(self):
        while not rospy.is_shutdown():
            file = self.path[self.idx]
            self.load_data(file)
            self.ros_rate.sleep()
            self.idx += 1
            if self.idx > len(self.path):
                self.idx = 0

         
    def load_data(self, file):
        self.load_hdf5_file(file)
        self.publish()




    # def load_data(self):
    #     for frame, file  in enumerate(self.path):
    #         # load file
    #         self.load_hdf5_file(file)
    #         self.publish()

    #         intensity_img = normalize(self.intensity_1)
    #         range_img = normalize(self.distance_m_1)
    #         if frame > self.max_frames:
    #             break

    #     print('### End of PointCloudDeNoising visualization')



    def get_rgb(self, labels):
        """returns color coding according to input labels """
        r = g = b = np.zeros_like(labels)
        
        for label_id, color in self.color_label_mapping.items():
            r = np.where(labels == label_id, color[0] / 255.0, r)
            g = np.where(labels == label_id, color[1] / 255.0, g)
            b = np.where(labels == label_id, color[2] / 255.0, b)
        return r, g, b

    def publish(self):
        """publish a single point cloud """
        if rospy is None:
            print('distance_m_1', self.distance_m_1.flatten())
            print('intensity_1', self.intensity_1.flatten())
            print('labels_1', self.labels_1.flatten())
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base'

        # http://wiki.ros.org/rviz/DisplayTypes/PointCloud
        r, g, b = self.get_rgb(self.labels_1.flatten())
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('distance', 12, PointField.FLOAT32, 1),
            PointField('intensity', 16, PointField.FLOAT32, 1),
            PointField('r', 20, PointField.FLOAT32, 1),
            PointField('g', 24, PointField.FLOAT32, 1),
            PointField('b', 28, PointField.FLOAT32, 1)
        ]

        points = list(zip(
            self.sensorX_1.flatten(),
            self.sensorY_1.flatten(),
            self.sensorZ_1.flatten(),
            self.distance_m_1.flatten(),
            self.intensity_1.flatten(),
            r, g, b
            ))

        cloud = pc2.create_cloud(header, fields, points)
        self.ros_publisher.publish(cloud)
        rospy.loginfo('PushPointCloud')
        



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

if __name__ == "__main__":
    print('### start PointCloudDeNoising visualization')
    data = RosPublisher()
    # rospy.spin()

# #!/usr/bin/python3
# """
# " author: Robin Heinzler
# " project: point cloud de-noising
# " date: 2020-02-05
# " info: visualization of point cloud data with rviz
# """
# import numpy as np
# import glob
# import h5py
# from numpy.core.arrayprint import printoptions
# from numpy.lib.function_base import select
# try:
#     import rospy
#     from sensor_msgs.msg import PointCloud2
#     import sensor_msgs.point_cloud2 as pc2
#     from sensor_msgs.msg import PointField
#     from std_msgs.msg import Header
# except ImportError:
#     print('ImportError Rospy!')
#     print('No visualization, but the data is shown in the terminal')
#     rospy = None

# # settings
# # change input path here
# PATH = "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/2018-11-29_145021_Static1-Day-Clear/"
# COLOR_LABEL_MAPPING = {
#     0: [0, 0, 0],       # no lable
#     100: [0, 0, 0],     # valid / clear
#     101: [255, 0, 0], # rain
#     102: [0, 0, 255], # fog
# }
# MAX_FRAMES = 200


# class RosPublisher:
#     def __init__(self, name='ros_publisher', color_label_mapping=COLOR_LABEL_MAPPING):
#         """initialize ros python api with bode 'RosPublisher' and set hdf5 channel names"""
#         self.color_label_mapping = color_label_mapping
#         # init ros
#         self.name = name
#         self.cloud_topic_name = "pointcloud"
#         if rospy is not None:
#             rospy.init_node(self.name + self.cloud_topic_name)
#             self.rostime = rospy.Time.now()
#             self.ros_rate = rospy.Rate(10)
#             self.ros_publisher = rospy.Publisher('RosPublisher/{}'.format(name), PointCloud2, queue_size=10)
#             self.r = rospy.Rate(10)

#         # define hdf5 data format
#         self.channels = ['labels_1', 'distance_m_1', 'intensity_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']
#         self.sensorX_1 = None
#         self.sensorY_1 = None
#         self.sensorZ_1 = None
#         self.distance_m_1 = None
#         self.intensity_1 = None
#         self.labels_1 = None
#         # 读入数据格式 array 32 * 400

#     def get_rgb(self, labels):
#         """returns color coding according to input labels """
#         r = g = b = np.zeros_like(labels)
#         for label_id, color in self.color_label_mapping.items():
#             r = np.where(labels == label_id, color[0] / 255.0, r)
#             g = np.where(labels == label_id, color[1] / 255.0, g)
#             b = np.where(labels == label_id, color[2] / 255.0, b)
#         return r, g, b

#     def publish(self):
#         """publish a single point cloud """
#         if rospy is None:
#             print('distance_m_1', self.distance_m_1.flatten())
#             print('intensity_1', self.intensity_1.flatten())
#             print('labels_1', self.labels_1.flatten())
#             return

#         header = Header()
#         header.stamp = rospy.Time.now()
#         header.frame_id = 'base'

#         # http://wiki.ros.org/rviz/DisplayTypes/PointCloud
#         r, g, b = self.get_rgb(self.labels_1.flatten())
#         fields = [
#             PointField('x', 0, PointField.FLOAT32, 1),
#             PointField('y', 4, PointField.FLOAT32, 1),
#             PointField('z', 8, PointField.FLOAT32, 1),
#             PointField('distance', 12, PointField.FLOAT32, 1),
#             PointField('intensity', 16, PointField.FLOAT32, 1),
#             PointField('r', 20, PointField.FLOAT32, 1),
#             PointField('g', 24, PointField.FLOAT32, 1),
#             PointField('b', 28, PointField.FLOAT32, 1)
#         ]

#         points = list(zip(
#             self.sensorX_1.flatten(),
#             self.sensorY_1.flatten(),
#             self.sensorZ_1.flatten(),
#             self.distance_m_1.flatten(),
#             self.intensity_1.flatten(),
#             r, g, b
#             ))

#         cloud = pc2.create_cloud(header, fields, points)
#         self.ros_publisher.publish(cloud)




#     def load_hdf5_file(self, filename):
#         """load one single hdf5 file with point cloud data

#         the coordinate system is based on the conventions for land vehicles (DIN ISO 8855)
#         (https://en.wikipedia.org/wiki/Axes_conventions)

#         each channel contains a matrix with 32x400 values, ordered in layers and columns
#         e.g. sensorX_1 contains the x-coordinates in a projected 32x400 view
#         """

#         with h5py.File(filename, "r", driver='core') as hdf5:
#             # for channel in self.channels:
#             self.sensorX_1 = hdf5.get('sensorX_1')[()]
#             self.sensorY_1 = hdf5.get('sensorY_1')[()]
#             self.sensorZ_1 = hdf5.get('sensorZ_1')[()]
#             self.distance_m_1 = hdf5.get('distance_m_1')[()]
#             self.intensity_1 = hdf5.get('intensity_1')[()]
#             self.labels_1 = hdf5.get('labels_1')[()]


# def main(path=PATH, max_frames=MAX_FRAMES):
#     """main function for reading and publishing the point cloud data"""
#     print('### start PointCloudDeNoising visualization')
#     pub = RosPublisher()

#     # get all files inside the defined dir
#     files = sorted(glob.glob(path + '*.hdf5'))
#     print('Directory {} contains are {} hdf5-files'.format(path, len(files)))

#     if len(files) == 0:
#         print('Please check the input dir {}. Could not find any hdf5-file'.format(path))
#     else:
#         print('Start publihsing the first {} frames...'.format(max_frames))
#         for frame, file in enumerate(files):
#             print('{:04d} / {}'.format(frame, file))

#             # load file
#             pub.load_hdf5_file(file)

#             # publish point cloud
#             pub.publish()

#             # stop condition
#             if frame == max_frames:
#                 break
#     print('### End of PointCloudDeNoising visualization')


# if __name__ == "__main__":
#     main()
