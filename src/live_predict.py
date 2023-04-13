import torch.nn as nn
import torch
import argparse
import os
import numpy as np
from weathnet import WeatherNet
from laserscan import LaserScan
import timeit
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visu import COLOR_LABEL_MAPPING
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        
else:
    DEVICE = torch.device('cpu')


class LivePredictor(object):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.color_label_mapping = COLOR_LABEL_MAPPING
        self.topics = ['/velodyne_points', '/livox/lidar', '/ouster1/os_cloud_node/points']
                
        if self.opt.data == 'velodyne16':
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 32, 416, 15, -15, 80
            topic = self.topics[0]
        
        elif self.opt.data == 'livox':
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 128, 300, 12.5, -12.5, 81.7
            topic = self.topics[1]
        
        elif self.opt.data == 'ouster':
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 64, 1024, 22.5, -22.5, 360
            topic = self.topics[2]

        self.laserscaner = LaserScan(project=True, H=self.H, W=self.W, fov_up=self.fov_up, fov_down=self.fov_down, hfov = self.hfov)
        self.range_img = self.laserscaner.proj_range 
        self.intensity_img = self.laserscaner.proj_remission
        self.xyz_img = self.laserscaner.proj_xyz

        self.load_model()

        rospy.init_node('LiDAR_Subscriber', anonymous=True)
        self.ros_publisher = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        self.lidar_sub = rospy.Subscriber(topic, PointCloud2, self.callback, queue_size=1)

    
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
        self.model.eval()

    
    def callback(self, lidar_msg):
        start = timeit.default_timer()
        np_p = []
        for point in pc2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            np_p.append(point[0])
            np_p.append(point[1])
            np_p.append(point[2])
            np_p.append(point[3])
                
        np_p = np.array(np_p).reshape((-1, 4))
        self.laserscaner.set_points(np_p[:,0:3], np_p[:, 3])

        # projected range image - [H, W]
        # projected intensity image - [H, W]
        # projected xyz image - [H, W, 3]
        self.range_img = self.laserscaner.proj_range 
        self.intensity_img = self.laserscaner.proj_remission
        self.xyz_img = self.laserscaner.proj_xyz

        stop1 = timeit.default_timer()
        print('projection_time: ', stop1 - start)

        image = self.load_data()

        predictions = self.do_prediction(image)

        self.publish_data(predictions, lidar_msg)
        
    
    def load_data(self):
        image = np.concatenate((self.range_img, self.intensity_img)).reshape((1, 2, self.H, self.W))
        return torch.from_numpy(image).type(torch.torch.FloatTensor)
    
    
    def do_prediction(self, data):
        with torch.no_grad():
                start = timeit.default_timer()

                images = data.to(DEVICE, dtype=torch.float)

                predictions = self.model(images)

                predictions = predictions.argmax(dim=1).squeeze().data.cpu()

                stop = timeit.default_timer()
                print('single inference: ', stop - start)
        
        return np.array(predictions).reshape((self.H, self.W))+100
    
    def publish_data(self, predictions, lidar_msg):
        """publish a single point cloud """
        if rospy is None:
            return

        # header = Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = 'base'
        header = lidar_msg.header

        # http://wiki.ros.org/rviz/DisplayTypes/PointCloud
        r, g, b = self.get_rgb(predictions.flatten())
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
            self.xyz_img[:, :, 0].flatten(),
            self.xyz_img[:, :, 1].flatten(),
            self.xyz_img[:, :, 2].flatten(),
            self.range_img.flatten(),
            self.intensity_img.flatten(),
            r, g, b
            ))

        cloud = pc2.create_cloud(header, fields, points)
        self.ros_publisher.publish(cloud)
        rospy.loginfo('PushPointCloud')

    def get_rgb(self, labels):
        """returns color coding according to input labels """
        r = g = b = np.zeros_like(labels)
        
        for label_id, color in self.color_label_mapping.items():
            r = np.where(labels == label_id, color[0] / 255.0, r)
            g = np.where(labels == label_id, color[1] / 255.0, g)
            b = np.where(labels == label_id, color[2] / 255.0, b)
        return r, g, b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='velodyne16')
    parser.add_argument(
        '--checkpoint', type=str, default='../checkpoints/saved_model.pth', help='checkpoint file')
    

    opt = parser.parse_args()
    LivePredictor(opt)
    rospy.spin()
