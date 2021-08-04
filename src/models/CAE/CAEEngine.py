import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary
from torchvision.utils import make_grid
from tqdm import tqdm

from image.ImageDifference import ImageDifference
from image.ImageElaboration import ImageElaboration
from image.RGB import RGB
from metrics.classification import accuracy, precision, recall, tpr, fpr
from utils import save_original_images, save_reconstructed_images, save_model, image_to_vid, save_loss_plot, \
    build_segmentation_plot


class CAEEngine:
    def __init__(self, net, trainloader, trainset, testloader, testset, epochs, optimizer, criterion, device):
        self.net = net
        self.trainloader = trainloader
        self.trainset = trainset
        self.testloader = testloader
        self.testset = testset

        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def start(self):
        best_val_loss = sys.maxsize
        best_epoch = 1

        # summary
        summary(self.net, (3, 256, 256))

        # a list to save all the reconstructed images in PyTorch grid format
        grid_images = []

        train_loss = []
        valid_loss = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1} of {self.epochs}")
            train_epoch_loss = self.train(
                self.net, self.trainloader, self.trainset, self.device, self.optimizer, self.criterion
            )
            valid_epoch_loss, recon_images, original_images = self.validate(
                self.net, self.testloader, self.testset, self.device, self.criterion
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            # save the reconstructed images from the validation loop
            save_reconstructed_images(recon_images, epoch + 1)
            save_original_images(original_images, epoch + 1)
            # save model
            save_model(self.net, epoch + 1)
            # convert the reconstructed images to PyTorch image grid format
            image_grid = make_grid(recon_images.detach().cpu())
            grid_images.append(image_grid)
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")

            if valid_epoch_loss < best_val_loss:
                best_val_loss = valid_epoch_loss
                best_epoch = epoch + 1

        # save the reconstructions as a .gif file
        image_to_vid(grid_images)
        # save the loss plots to disk
        save_loss_plot(train_loss, valid_loss)
        print('TRAINING COMPLETE')

        return self.net, best_epoch

    @staticmethod
    def train(model, dataloader, dataset, device, optimizer, criterion):
        model.train()
        running_loss = 0.0
        counter = 0
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            data = data['image']
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        train_loss = running_loss / (counter * dataloader.batch_size)
        return train_loss

    @staticmethod
    def validate(model, dataloader, dataset, device, criterion):
        model.eval()
        running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
                counter += 1
                data = data['image']
                data = data.to(device)
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                running_loss += loss.item()

                # save the last batch input and output of every epoch
                if i == int(len(dataloader) / dataloader.batch_size) - 1:
                    recon_images = reconstruction
                    original_images = data
        val_loss = running_loss / counter
        return val_loss, recon_images, original_images

    def segmentation_performance_computation(self, net, testloader, testset, threshold):
        counter = 0
        for i, data in tqdm(enumerate(testloader), total=len(testset)):
            counter += 1
            img = data['image']
            img = img.to(self.device)
            reconstruction = net(img)
            loss = self.criterion(reconstruction, img)
            loss.backward()

            if loss > threshold:
                diff = ImageDifference(img, reconstruction).difference()
                elaborated = ImageElaboration(diff)
                elaborated.keep_only(RGB.RED)
                elaborated.negative()
                build_segmentation_plot(img, reconstruction, diff, elaborated, counter)

    def classification_performance_computation(self, net, testloader, testset, thresholds):
        max_acc = 0
        best_th = 0

        res = []
        for i, data in tqdm(enumerate(testloader), total=len(testset)):
            img = data['image']
            img = img.to(self.device)
            reconstruction = net(img)
            loss = self.criterion(reconstruction, img)
            loss.backward()

            res.append({'loss': loss, 'realLabel': data['label']})

        for t in thresholds:
            acc, pre, rec, tp_rate, fp_rate = self.classify(res, t)

            if acc > max_acc:
                max_acc = acc
                best_th = t

        return best_th

    def visualization(self, net, testset, slot_num=2):
        net.eval()

        plt.figure()

        with torch.no_grad():
            for i in range(len(testset)):
                sample = testset[i]['image']

                reconstruction = net(sample[None, ...])

                loss = self.criterion(reconstruction, sample[None, ...])

                ax = plt.subplot(1, slot_num * 2, (i * 2) + 1)
                plt.tight_layout()
                ax.set_title('Original'.format(i + 1))
                ax.axis('off')
                plt.imshow(np.transpose(sample, (1, 2, 0)))

                ax = plt.subplot(1, slot_num * 2, (i * 2) + 2)
                plt.tight_layout()
                ax.set_title('Recon error: {}'.format(loss))
                ax.axis('off')
                plt.imshow(np.transpose(np.squeeze(reconstruction), (1, 2, 0)))

                if i == (slot_num - 1):
                    plt.show()
                    break

    @staticmethod
    def classify(res, threshold):
        data = []
        for r in res:
            label = 0 if r['loss'] < threshold else 1
            data.append({'label': label, 'realLabel': r['realLabel']})

        acc = accuracy(data)
        pre = precision(data)
        rec = recall(data)
        tp_rate = tpr(data)
        fp_rate = fpr(data)

        print('--------- Threshold: ' + str(threshold) + ' ----------')
        print('Accuracy: ' + str(acc))
        print('Precision: ' + str(pre))
        print('Recall: ' + str(rec))
        print('TPR: ' + str(tp_rate))
        print('FPR' + str(fp_rate))

        return acc, pre, rec, tp_rate, fp_rate
