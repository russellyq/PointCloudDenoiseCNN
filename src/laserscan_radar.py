from re import S
from signal import set_wakeup_fd
from PIL import Image
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import rospy
import matplotlib.pyplot as plt
import h5py
import cv2
import timeit
from cv_bridge import CvBridge, CvBridgeError
import argparse

def array2h5df(array_x, array_y, array_z, array_d, array_i, file):
  with h5py.File(file, 'w') as hf:
    hf.create_dataset('sensorX_1', data=array_x)
    hf.create_dataset('sensorY_1', data=array_y)
    hf.create_dataset('sensorZ_1', data=array_z)
    hf.create_dataset('distance_m_1', data=array_d)
    hf.create_dataset('intensity_1', data=array_i)


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin', '.npz']

  def __init__(self, project=True, H=16, W=200, fov_up=15, fov_down=-15, hfov = 85.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.hfov = hfov
    self.reset()

    # self.lidar_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.lidar_callback(), queue_size=1)

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    # scan = np.fromfile(filename, dtype=np.float32)
    scan = np.load(filename)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)


    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    hfov_rad = self.hfov / 180 * np.pi ### Preset HFOV
    # get projections in image coords
    # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x = 0.5 * (yaw / (hfov_rad / 2.0) + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]

    # x = self.points[:, 0]
    # y = self.points[:, 1]
    # z = self.points[:, 2]

    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices

    self.proj_mask = (self.proj_idx > 0).astype(np.int32)

class PointCloudProjection(object):
  def __init__(self, opt) -> None:
    super().__init__()

    self.opt = opt

    self.number = 0
    self.topics = ['/velodyne_points', '/livox/lidar', '/livox/lidar/time_sync', '/radar/radar_pc']
    self.bridge = CvBridge()
    print(opt.data)

    
    if self.opt.data == 'velodyne16':
            print('velodyne16 !')
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 16, 416, 15, -15, 80
            topic = self.topics[0]
        
    elif self.opt.data == 'livox':
            print('livox !')
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 64, 400, 12.5, -12.5, 81.7
            topic = self.topics[1]
    
    elif self.opt.data == 'radar':
            print('radar !')
            self.H, self.W, self.fov_up, self.fov_down, self.hfov = 320, 320, 15, -15, 90
            topic = self.topics[3]
    
    self.laserscaner = LaserScan(project=True, H=self.H, W=self.W, fov_up=self.fov_up, fov_down=self.fov_down, hfov = self.hfov) 
    
    self.range_img = self.laserscaner.proj_range 
    self.intensity_img = self.laserscaner.proj_remission
    self.xyz_img = self.laserscaner.proj_xyz

    rospy.init_node('LiDAR_Subscriber', anonymous=True)
    self.lidar_sub = rospy.Subscriber(topic, PointCloud2, self.callback, queue_size=99999)

    self.range_img_pub = rospy.Publisher('/range_img', Image, queue_size=1)
    self.intensity_img_pub = rospy.Publisher('/intensity_img', Image, queue_size=1)
    


  def callback(self, lidar_msg):
    self.number += 1
    start = timeit.default_timer()
    np_p = []
    if self.opt.data == 'radar':
      for point in pc2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z", "RCS", "v_r", "v_r_compensated")):
        if point[0] == 0 and point[1] == 0 and point[2] == 0:
          continue
        np_p.append(point[0])
        np_p.append(point[1])
        np_p.append(point[2])
        np_p.append(point[3])
        # np_p.append(point[4])
        # np_p.append(point[5])
    else:
          for point in pc2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
              continue
            np_p.append(point[0])
            np_p.append(point[1])
            np_p.append(point[2])
            np_p.append(point[3])

            
    np_p = np.array(np_p).reshape((-1, 4))
    self.laserscaner.set_points(np_p[:,0:3], np_p[:, 3])

    # projected range image - [H,W]
    # projected intensity image - [H,W]
    # projected xyz image - [H,W]
    self.range_img = self.laserscaner.proj_range 
    self.intensity_img = self.laserscaner.proj_remission
    self.xyz_img = self.laserscaner.proj_xyz

    stop = timeit.default_timer()
    print('projection_time: ', stop - start)
    print(self.range_img.shape)
    print(self.intensity_img.shape)
    print(self.xyz_img.shape)

    range_imgmsg = self.bridge.cv2_to_imgmsg(cv2.applyColorMap(normalize(self.range_img), cv2.COLORMAP_JET), 'passthrough')
    intensity_imgmsg = self.bridge.cv2_to_imgmsg(cv2.applyColorMap(normalize(self.intensity_img), cv2.COLORMAP_JET), 'passthrough')
    range_imgmsg.header, intensity_imgmsg.header = lidar_msg.header, lidar_msg.header 
    self.range_img_pub.publish(range_imgmsg)
    self.intensity_img_pub.publish(intensity_imgmsg)

    # file_name = '/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/our_data/livox/test_' + str(self.number) + '.hdf5'
    # array2h5df(self.xyz_img[:, :, 0], self.xyz_img[:, :, 1], self.xyz_img[:, :, 2], self.range_img, self.intensity_img, file_name)




def normalize(x):
    return ( 255 * (x-x.min()) / (x.max()-x.min()) ).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='radar')
    opt = parser.parse_args()
    print(opt.data)
    projection = PointCloudProjection(opt)
    rospy.spin()
#     # file_name = "mp_data.npz"
#     # converter = LaserScan(project=True, H=16, W=200, fov_up=15, fov_down=-15, hfov = 360)
#     # converter.open_scan(file_name)
#     # depth_img = converter.proj_range 
#     # intensity_img = converter.proj_remission
#     # xyz_img = converter.proj_xyz

#     # ## Display ####
#     # plt.figure()
#     # plt.subplot(2, 1, 1)
#     # plt.imshow(depth_img, cmap="gray")
#     # plt.subplot(2, 1, 2)
#     # plt.imshow(intensity_img, cmap="gray")
#     # plt.show()


#     # cv2.imshow('depth_img', cv2.applyColorMap(normalize(depth_img), cv2.COLORMAP_JET))
#     # cv2.waitKey(0)
#     # cv2.imshow('intensity_img', cv2.applyColorMap(normalize(intensity_img), cv2.COLORMAP_JET))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
