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
from utils import mIoU
from torch.autograd import Variable
from torchvision.transforms import transforms

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        
else:
    DEVICE = torch.device('cpu')

result_path= './result.txt'
if os.path.exists(result_path):
    os.remove(result_path)

def main(opt):

    # lodar dataset & dataloader
    train_dataset = DeNoiseDataset(mode='train')
    test_dataset = DeNoiseDataset(mode='test')
    val_dataset = DeNoiseDataset(mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    # network
    model = WeatherNet()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(DEVICE)

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-18, betas=(0.9, 0.999), eps=1e-8)

    min_valid_loss = np.inf

    for epoch in range(opt.epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        # train
        for i, data in enumerate(train_dataloader, 0):
            images, labels = data
            images, labels = Variable(images.to(DEVICE, dtype=torch.float)), Variable(labels.to(DEVICE, dtype=torch.long))
            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()*labels.size(0)

            if i % 100 == 99:
                print("epoch:%d, %d/%d, train_loss:%0.3f" % (epoch+1, i+1, (len(train_dataset) - 1) // opt.batch_size + 1, loss.cpu().item()*labels.size(0)))
        
        train_loss /= len(train_dataset)
        print('\nepoch:{}, train_loss:{:.4f}'.format(epoch+1, train_loss))
        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, train_loss:{:.4f}'.format(epoch+1, train_loss))
            f.close()

        # validation
        model.eval()
        for i, data in enumerate(val_dataloader, 0):
            images, labels = data
            images, labels = images.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels)
            valid_loss += loss.cpu().item()*labels.size(0)

            if i % 100 ==99:
                print("epoch:%d, %d/%d, val_loss:%0.3f" % (epoch+1, i+1, (len(val_dataset) - 1) // opt.batch_size + 1, loss.cpu().item()*labels.size(0)))

        valid_loss /= len(val_dataset)
        print('\nepoch:{}, val_loss:{:.4f}'.format(epoch+1, valid_loss))
        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, val_loss:{:.4f}'.format(epoch+1, valid_loss))
            f.close()
        
        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            save_name = '../checkpoints/saved_model_' +str(epoch) + '_.pth'
            torch.save(model.state_dict(), save_name)

        # # test
        # clear_mIoU, rain_mIoU, fog_mIoU = 0.0, 0.0, 0.0
        # number = 0
        # with torch.no_grad():
        #     for i, data in enumerate(test_dataloader, 0):
        #         images, labels = data
        #         images, labels = images.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)

        #         predictions = model(images)

        #         labels = labels.cpu().detach().numpy()
        #         predictions = predictions.cpu().detach().numpy()

        #         for prediction, label in zip(predictions, labels):
        #             clear_mIoU += mIoU(prediction, label, 'clear')
        #             rain_mIoU += mIoU(prediction, label, 'rain')
        #             fog_mIoU += mIoU(prediction, label, 'fog')
        #             number += 1

        # clear_mIoU, rain_mIoU, fog_mIoU = clear_mIoU / number, rain_mIoU / number, fog_mIoU / number
        # print('[%d] acc: %.3f, %.3f, %.3f' % (epoch + 1, clear_mIoU, rain_mIoU, fog_mIoU))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=20, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=50, help='number of epochs to train for')
    

    opt = parser.parse_args()
    main(opt)

