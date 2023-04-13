import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
from dataset_utils import DeNoiseDataset
from weathnet import WeatherNet
from torch.autograd import Variable
from torchvision.transforms import transforms
from utils import label_accuracy_score, Evaluator
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        
else:
    DEVICE = torch.device('cpu')



def test(opt):
    test_result_path= './result_test.txt'
    if os.path.exists(test_result_path):
        os.remove(test_result_path)

    # lodar dataset & dataloader

    test_dataset = DeNoiseDataset(mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    print('number of test data:', len(test_dataset))
    
    # network
    model = WeatherNet()
    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint)
    print('Loading saved model !')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    model.to(DEVICE)
    model.eval()

    test_label_true = torch.LongTensor()
    test_label_pred = torch.LongTensor()
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            images, labels = data
            images, labels = images.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)

            predictions = model(images)

            # labels = labels.cpu().detach().numpy()
            # predictions = predictions.argmax(dim=1).squeeze().data.cpu().detach().numpy()
            # predictions = predictions.astype('int')
            labels = labels.cpu()
            predictions = predictions.argmax(dim=1).squeeze().data.cpu()

            test_label_true = torch.cat((test_label_true, labels), dim=0)
            test_label_pred = torch.cat((test_label_pred, predictions), dim=0)

        #     evaluator = Evaluator(3)
        #     evaluator.add_batch(labels.detach().numpy(), predictions.detach().numpy().astype('int'))
              
        # Acc = evaluator.Pixel_Accuracy()
        # Acc_class = evaluator.Pixel_Accuracy_Class()
        # mIoU = evaluator.Mean_Intersection_over_Union()
        # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        # print("Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))
        
        val_acc, val_acc_cls, iu, val_fwavacc = label_accuracy_score(test_label_true.numpy(), test_label_pred.numpy(), 3)
        print('IoU: {:.4f}, {:.4f}, {:.4f}'.format(iu[0], iu[1], iu[2]))
        val_mean_iu = np.nanmean(iu)
        print('Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwavacc:{:.4f}'.format(val_acc, val_acc_cls, val_mean_iu, val_fwavacc))
        with open(test_result_path, 'a') as f:
            f.write(str(datetime.datetime.now()))
            f.write('\n number of test data: {:d}\n'.format(len(test_dataset)))
            f.write('\n IoU: {:.4f}, {:.4f}, {:.4f}\n'.format(iu[0], iu[1], iu[2]))
            f.write('\n Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwavacc:{:.4f}\n'.format(val_acc, val_acc_cls, val_mean_iu, val_fwavacc))
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=20, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument(
        '--checkpoint', type=str, default='../checkpoints/saved_model_50.pth', help='checkpoint file')
    

    opt = parser.parse_args()
    test(opt)

