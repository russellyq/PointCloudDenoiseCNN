import numpy as np
import math

CONFIG = {
    'fov_up': 20,
    'fov_down': -20,
    'number': 32,
    'img_width': 512
}

LABEL_CLASS_MAPPING = {
    'clear': 0,           # valid / clear  -> 0
    'rain': 1, # rain             -> 1
    'fog': 2, # fog              -> 2
}

fov_up_rad = CONFIG['fov_up'] / 180 * math.pi
fov_down_rad = CONFIG['fov_down'] / 180 * math.pi
fov_rad = abs(fov_up_rad) + abs(fov_down_rad) 

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

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

def mIoU(preds, labels, class_name='clear'):
    intersction = 0
    union = 0
    label = LABEL_CLASS_MAPPING[class_name]
    for i, j in zip(preds.flatten(), labels.flatten()):
        print(i, j)
        if i == label and j == label:
            intersction += 1
        if i == label or j == label:
            union +=1
    if union == 0:
        return 0
    return intersction / union



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
    return acc, acc_cls, mean_iu, fwavacc