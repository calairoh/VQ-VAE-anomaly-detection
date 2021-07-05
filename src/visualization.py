import matplotlib.pyplot as plt
import numpy as np
import random

import torch


def visualization(net, test_loader, device):
    net.eval()
    with torch.no_grad():
        for data in random.sample(list(test_loader), 5):
            imgs, _ = data
            imgs = imgs.to(device)
            img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            out, mu, log_var = net(imgs)
            outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()
            break
