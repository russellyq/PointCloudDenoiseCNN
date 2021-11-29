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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        
else:
    DEVICE = torch.device('cpu')


def main(opt):

    # lodar dataset & dataloader
    train_dataloader = DataLoader(DeNoiseDataset(mode='train'), batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(DeNoiseDataset(mode='test'), batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(DeNoiseDataset(mode='val'), batch_size=opt.batch_size, shuffle=True)

    # network
    model = WeatherNet()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(DEVICE)

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-18, betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(opt.epochs):
        running_loss = 0.0
        valid_loss = 0.0
        min_valid_loss = np.inf

        # train
        for i, data in enumerate(train_dataloader, 0):
            images, labels = data
            images, labels = Variable(images.to(DEVICE, dtype=torch.float)), Variable(labels.to(DEVICE, dtype=torch.long))
            optimizer.zero_grad()

            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

        # validation
        model.eval()
        for i, data in enumerate(val_dataloader, 0):
            images, labels = data
            images, labels = images.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()

            prediction = model(images)
            loss = criterion(prediction, labels)
            valid_loss += loss.item()

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                save_name = 'saved_model_' +str(epoch) + '_.pth'
                torch.save(model.state_dict(), save_name)

        # test
        clear_mIoU, rain_mIoU, fog_mIoU = 0.0, 0.0, 0.0
        number = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                images, labels = data
                images, labels = images.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)

                prediction = model(images)
                labels = labels.cpu().detach().numpy()
                predictions = predictions.cpu().detach().numpy()

                for image, prediction in zip(images, predictions):
                    clear_mIoU += mIoU(image, prediction, 'clear')
                    rain_mIoU += mIoU(image, prediction, 'rain')
                    fog_mIoU += mIoU(image, prediction, 'fog')
                    number += 1
        clear_mIoU, rain_mIoU, fog_mIoU = clear_mIoU / number, rain_mIoU / number, fog_mIoU / number
        print('[%d] acc: %.3f, %.3f, %.3f' % (epoch + 1, clear_mIoU, rain_mIoU, fog_mIoU))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--epochs', type=int, default=1, help='number of epochs to train for')
    

    opt = parser.parse_args()
    main(opt)

