import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
import torch
import random
import numpy as np
# from torchvision import transforms
# from PIL import Image, ImageOps, ImageFilter
# from scipy.ndimage.filters import gaussian_filter
from utils import get_file_list
TRAIN_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/train_01/",
                       "/home/newdisk/yanqiao/dataset/cnn_denoising/train_02/"]
                    #    "/home/newdisk/yanqiao/dataset/cnn_denoising/train_road_01/",
                    #    "/home/newdisk/yanqiao/dataset/cnn_denoising/train_road_02/"]
TEST_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/test_01/"]
VAL_ROOT_DIR_LIST = ["/home/newdisk/yanqiao/dataset/cnn_denoising/val_01/"]


# TRAIN_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_01/",
#                        "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_02/",
#                        "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_01/",
#                        "/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/train_road_02/"]
# TEST_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/test_01/"]
# VAL_ROOT_DIR_LIST = ["/media/yq-robot/Seagate Backup Plus Drive/dataset/cnn_denoise/cnn_denoising/val_01/"]



def normalize(x):
    return (255*x).astype(np.uint8) if x.max()==x.min() else ( 255 * (x-x.min()) / (x.max()-x.min()) ).astype(np.uint8)



# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """
#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         img = np.array(img).astype(np.float32)
#         mask = np.array(mask).astype(np.float32)
#         img /= 255.0
#         img -= self.mean
#         img /= self.std

#         return {'image': img,
#                 'label': mask}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         img = sample['image']
#         mask = sample['label']
#         img = np.array(img).astype(np.float32).transpose((2, 0, 1))
#         mask = np.array(mask).astype(np.float32)

#         img = torch.from_numpy(img).float().type(torch.torch.FloatTensor)
#         mask = torch.from_numpy(mask).float().type(torch.torch.FloatTensor)

#         return {'image': img,
#                 'label': mask}


# class RandomHorizontalFlip(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         if random.random() < 0.5:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

#         return {'image': img,
#                 'label': mask}


# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         rotate_degree = random.uniform(-1*self.degree, self.degree)
#         img = img.rotate(rotate_degree, Image.BILINEAR)
#         mask = mask.rotate(rotate_degree, Image.NEAREST)

#         return {'image': img,
#                 'label': mask}


# class RandomGaussianBlur(object):
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         if random.random() < 0.5:
#             # img = img.filter(ImageFilter.GaussianBlur(
#             #     radius=random.random()))
#             img = gaussian_filter(img, sigma=7)

#         return {'image': img,
#                 'label': mask}


# class RandomScaleCrop(object):
#     def __init__(self, base_size, crop_size, fill=0):
#         self.base_size = base_size
#         self.crop_size = crop_size
#         self.fill = fill

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         # random scale (short edge)
#         short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
#         w, h = img.size
#         if h > w:
#             ow = short_size
#             oh = int(1.0 * h * ow / w)
#         else:
#             oh = short_size
#             ow = int(1.0 * w * oh / h)
#         img = img.resize((ow, oh), Image.BILINEAR)
#         mask = mask.resize((ow, oh), Image.NEAREST)
#         # pad crop
#         if short_size < self.crop_size:
#             padh = self.crop_size - oh if oh < self.crop_size else 0
#             padw = self.crop_size - ow if ow < self.crop_size else 0
#             img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
#             mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
#         # random crop crop_size
#         w, h = img.size
#         x1 = random.randint(0, w - self.crop_size)
#         y1 = random.randint(0, h - self.crop_size)
#         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

#         return {'image': img,
#                 'label': mask}


# class FixScaleCrop(object):
#     def __init__(self, crop_size):
#         self.crop_size = crop_size

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         w, h = img.size
#         if w > h:
#             oh = self.crop_size
#             ow = int(1.0 * w * oh / h)
#         else:
#             ow = self.crop_size
#             oh = int(1.0 * h * ow / w)
#         img = img.resize((ow, oh), Image.BILINEAR)
#         mask = mask.resize((ow, oh), Image.NEAREST)
#         # center crop
#         w, h = img.size
#         x1 = int(round((w - self.crop_size) / 2.))
#         y1 = int(round((h - self.crop_size) / 2.))
#         img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
#         mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

#         return {'image': img,
#                 'label': mask}

# class FixedResize(object):
#     def __init__(self, size):
#         self.size = (size, size)  # size: (h, w)

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']

#         assert img.size == mask.size

#         img = img.resize(self.size, Image.BILINEAR)
#         mask = mask.resize(self.size, Image.NEAREST)

#         return {'image': img,
#                 'label': mask}
        
        
class DeNoiseDataset(Dataset):
    def __init__(self, mode):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
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
        # label = labels_1.astype(np.uint8)
        label = labels_1

        return torch.from_numpy(image).type(torch.torch.FloatTensor), torch.from_numpy(label).type(torch.torch.FloatTensor)

        # sample = {'image': image,
        #             'label': label}

        # if self.mode == 'train':
        #     composed_transforms = transforms.Compose([
        #         RandomHorizontalFlip(),
        #         RandomGaussianBlur(),
        #         ToTensor()
        #     ])
        #     return composed_transforms(sample)
        # else:
        #     sample = {'image': torch.from_numpy(image).type(torch.torch.FloatTensor),
        #           'label': torch.from_numpy(label).type(torch.torch.FloatTensor)}
        #     return sample
    
    def load_hdf5_file(self, filename):
        # each channel contains a matrix with 32x400 values, ordered in layers and columns
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
        # distance_m_1 = normalize(distance_m_1)
        # intensity_1 = normalize(intensity_1)

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