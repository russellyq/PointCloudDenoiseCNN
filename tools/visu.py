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
from utils import normalize
import cv2


PATH = "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/2018-11-29_152024_Static1-Day-FogC/"
COLOR_LABEL_MAPPING = {
    0: [0, 0, 0],       # no lable
    100: [0, 0, 0],     # valid / clear
    101: [255, 0, 0], # rain
    102: [0, 0, 255], # fog
}
MAX_FRAMES = 10

class RosPublisher(object):
    def __init__(self, color_label_mapping=COLOR_LABEL_MAPPING, path=PATH, max_frames=MAX_FRAMES) -> None:
        super().__init__()
        """initialize ros python api with bode 'RosPublisher' and set hdf5 channel names"""
        self.color_label_mapping = color_label_mapping
        self.path = path
        self.max_frames = max_frames

        rospy.init_node('ros_publisher_pointcloud')
        self.rostime = rospy.Time.now()
        self.ros_rate = rospy.Rate(10)
        self.ros_publisher = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        self.r = rospy.Rate(10)


        # define hdf5 data format
        self.channels = ['labels_1', 'distance_m_1', 'intensity_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']
        self.sensorX_1 = None
        self.sensorY_1 = None
        self.sensorZ_1 = None
        self.distance_m_1 = None
        self.intensity_1 = None
        self.labels_1 = None

        fps, w, h = 2, 400, 32
        save_path = '../intensity.avi'
        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (w, h))

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

                intensity_img = normalize(self.intensity_1)
                range_img = normalize(self.distance_m_1)

                
                self.vid_writer.write(np.uint8(255*intensity_img))
            self.vid_writer.release()
        print('### End of PointCloudDeNoising visualization')



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
