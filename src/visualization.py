import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn

from src.engine import final_loss


def visualization(net, testset, slot_num=2):
    net.eval()

    plt.figure()

    with torch.no_grad():
        for i in range(len(testset)):
            sample = testset[i]['image']

            reconstruction, mu, logvar = net(sample[None, ...])

            criterion = nn.BCELoss(reduction='sum')
            bce_loss = criterion(reconstruction, sample[None, ...])
            loss = final_loss(bce_loss, mu, logvar)

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
