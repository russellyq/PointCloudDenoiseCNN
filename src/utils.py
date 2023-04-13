import numpy as np
import math
import os

CONFIG = {
    'fov_up': 20,
    'fov_down': -20,
    'number': 32,
    'img_width': 512
}
fov_up_rad = CONFIG['fov_up'] / 180 * math.pi
fov_down_rad = CONFIG['fov_down'] / 180 * math.pi
fov_rad = abs(fov_up_rad) + abs(fov_down_rad) 

def normalize(x):
    return x if x.max()==x.min() else (x - x.min()) / (x.max() - x.min())

def get_file_list(root_dir_list):
    file_list = []
    for root_dir in root_dir_list:
        for dir, root, files in os.walk(root_dir):
            for file in files:
                file_list.append(dir+'/'+file)
    return file_list

def lidar_to_image(X_list, Y_list, Z_list, Distance, Intensity):
    # input lidar points: array
    # X, Y, Z, Intensity: 32*400

    d_max = max(Distance)
    i_max = max(Intensity)

    range_img = np.zeros((CONFIG['number'], CONFIG['img_width']), dtype=np.float)
    intensity_img = np.zeros((CONFIG['number'], CONFIG['img_width']), dtype=np.float)

    for x, y, z, d, intensity in zip(X_list, Y_list, Z_list, Distance, Intensity):
        if d != 0:
            # range = math.sqrt(x**2+y**2+z**2)
            range = d
            
            yaw = math.atan2(y, x)
            pitch = math.asin(z / range)
            
            u = (0.5 * (1 + yaw / math.pi) ) * CONFIG['img_width']
            v = ( 1 - ( pitch +  abs(fov_down_rad) ) / fov_rad) * CONFIG['number']
            
            u = min(CONFIG['img_width'] - 1, math.floor(u))
            u = max(0, u)
            pixel_u = int(u)

            v = min(CONFIG['number'] -1, math.floor(v))
            v = max(0, v)
            pixel_v = int(v)

            # print(pixel_v, pixel_u, range, intensity)

            range_img[pixel_v, pixel_u] = range / d_max
            print(intensity, i_max)
            intensity_img[pixel_v, pixel_u] = intensity / i_max

    return (255*range_img).astype(np.uint8), (255*intensity_img).astype(np.uint8)
    # return 255*normalize(range_img).astype(np.uint8), 255*normalize(intensity_img).astype(np.uint8)



#得到混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

#计算图像分割衡量系数
def label_accuracy_score(label_trues, label_preds, n_class):
    """
     :param label_preds: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数
     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues,label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()

    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / ( hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) )
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # return acc, acc_cls, mean_iu, fwavacc
    return acc, acc_cls, iu, fwavacc

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # return all class overall pixel accuracy
		# """
		# 1、像素准确率(PA)
		# 像素准确率是所有分类正确的像素数占像素总数的比例。利用混淆矩阵计算则为（对角线元素之和除以矩阵所有元素之和）：
		# """
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        #返回平均精准率
        #注意，np.nanmean会忽略为空的类别
		# """
		# 2、平均像素准确率(MPA)
		# 	平均像素准确率是分别计算每个类别分类正确的像素数占所有预测为该类别像素数的比例，即精确率，然后累加求平均。
		# 利用混淆矩阵计算公式为(每一类的精确率Pi都等于对角线上的TP除以对应类别的像素数) ：
		# """
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # """
		# 3、平均交并比：
		# 平均交并比是对每一类预测的结果和真实值的交集与并集的比值求和平均的结果
		
		# """
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
		# """
		# 4、 频权交并比(FWloU)
		# 　频权交并比是根据每一类出现的频率设置权重，权重乘以每一类的IoU并进行求和，
		# 利用混淆矩阵计算：每个类别的真实数目为TP+FN，总数为TP+FP+TN+FN，其中每一类的权重和其IoU的乘积计算公式如下，在将所有类别的求和即可：
		# """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)#计算各个类别的数量矩阵
        
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


