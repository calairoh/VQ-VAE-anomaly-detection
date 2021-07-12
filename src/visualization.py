import matplotlib.pyplot as plt
import numpy as np
import random

import torch


def visualization(net, testset):
    net.eval()

    plt.figure()

    with torch.no_grad():
        for i in range(len(testset)):
            sample = testset[i]

            #sample = np.transpose(sample, (1, 2, 0))

            reconstruction, mu, logvar = net(sample[None, ...])

            ax = plt.subplot(1, 4, (i * 2) + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(np.transpose(sample, (1, 2, 0)))

            ax = plt.subplot(1, 4, (i * 2) + 2)
            plt.tight_layout()
            ax.set_title('Reconstruction #{}'.format(i))
            ax.axis('off')
            plt.imshow(np.transpose(np.squeeze(reconstruction), (1, 2, 0)))

            if i == 1:
                plt.show()
                break
