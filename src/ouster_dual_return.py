import os
import numpy as np
import timeit
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
import message_filters
import matplotlib.pyplot as plt
import open3d
import sensor_msgs.point_cloud2 as pcl2
import struct

class LivePredictor(object):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.topics = ['/os_cloud_node/points', '/os_cloud_node/points2']
        self.ratio = []
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('rgba', 12, PointField.UINT32, 1),
                        ]
        rospy.init_node('LiDAR_Subscriber', anonymous=True)
        rospy.loginfo("ROS node started")
        self.ros_publisher = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        self.lidar_sub1 = message_filters.Subscriber(self.topics[0], PointCloud2)
        self.lidar_sub2 = message_filters.Subscriber(self.topics[1], PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub1, self.lidar_sub2], 1, 0.1)
        self.ts.registerCallback(self.callback)
    

    
    def callback(self, lidar_msg1, lidar_msg2):
        header = lidar_msg1.header
        
        start = timeit.default_timer()
        rospy.loginfo("Recieving MSG")
        
        pointcloud_array1 = []
        for point in pc2.read_points(lidar_msg1, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            pointcloud_array1.append(point[0])
            pointcloud_array1.append(point[1])
            pointcloud_array1.append(point[2])
                
        pointcloud_array1 = np.array(pointcloud_array1).reshape((-1, 3))
        
        pointcloud_array2 = []
        for point in pc2.read_points(lidar_msg2, skip_nans=True, field_names=("x", "y", "z")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            pointcloud_array2.append(point[0])
            pointcloud_array2.append(point[1])
            pointcloud_array2.append(point[2])
                
        pointcloud_array2 = np.array(pointcloud_array2).reshape((-1, 3))
        
        ratio = pointcloud_array2.shape[0] / pointcloud_array1.shape[0]
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.concatenate((pointcloud_array1, pointcloud_array2),axis=0))
        
        self.ratio.append(ratio)
        print('ratio: (ouster2 / ouster1): ', ratio)
        # return None
        
        pointcloud_array = []
        
        if ratio < 0.05:
            rospy.loginfo("******************** No Smoke detected *******************")
            idx = []
            rospy.loginfo("\n")
            
        else:
            rospy.loginfo("---------------------- Smoke detected --------------------")
            
            pcd1 = open3d.geometry.PointCloud()
            pcd1.points = open3d.utility.Vector3dVector(pointcloud_array1)
            
            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(pointcloud_array2)
            
            index_i, query = self.find_closest_cluster(pcd2)
            # display_inlier_outlier(pcd2, index_i)
            
            
            pcd_kdtree = open3d.geometry.KDTreeFlann(pcd)
            [k, idx, _] = pcd_kdtree.search_hybrid_vector_3d(query[0:3], radius=2, max_nn=pointcloud_array2.shape[0]*2)
            
        for id, pcd_point in enumerate(np.asarray(pcd.points)):
            if id in idx:
                r = 255
                g = 0
                b = 0
            else:
                r = 255
                g = 255
                b = 255
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pointcloud_array.append([pcd_point[0], pcd_point[1], pcd_point[2], rgb])
        
        pointcloud_msg = pcl2.create_cloud(header, self.fields, pointcloud_array)
        self.ros_publisher.publish(pointcloud_msg)
            # print(k)
            # display_inlier_outlier(pcd, idx)

    

    def find_closest_cluster(self, pcd2):
        labels = np.array(pcd2.cluster_dbscan(eps=self.opt.eps2, min_points=self.opt.min_points2))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # pcd2.colors = open3d.utility.Vector3dVector(colors[:, :3])
        # open3d.visualization.draw_geometries([pcd2])
        
        indx = []
        min_dis = 9999
        for i in range(0, max_label+1):
            index_i = np.where(labels == i)[0].tolist()
            pcd_i = pcd2.select_by_index(index_i, invert=False)
            
            dis_matrix = np.linalg.norm(np.asarray(pcd_i.points), axis=1).reshape((-1, 1))
            
            dis = np.median(dis_matrix)
            if dis <= min_dis:
                min_dis = dis
                indx = index_i
                points_array = np.concatenate([np.asarray(pcd_i.points), dis_matrix], axis=1)
                points_array = points_array[points_array[:, 3].argsort()]
        return indx, points_array[len(points_array)//2-1]
    
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--eps1', type=float, default=0.2)
    # parser.add_argument(
    #     '--min_points1', type=int, default=10)
    parser.add_argument(
        '--eps2', type=float, default=0.1)
    parser.add_argument(
        '--min_points2', type=int, default=5) 
    opt = parser.parse_args()
    pre = LivePredictor(opt)
    while not rospy.is_shutdown():
        rospy.spin()
        ratio=pre.ratio
    # # print(ratio)
    # x_label = [i/10 for i in range(len(ratio))]
    # plt.plot(x_label, ratio)
    # plt.xlabel('time(s)')
    # plt.ylabel('ratio')
    # plt.title('dual / single')
    # plt.savefig('ratio_2022-08-19.png')
