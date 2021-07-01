import torch
import torch.nn.functional as F


def training(VAE, device, train_loader):
    """
    Initialize Hyperparameters
    """
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10

    """
    Initialize the network and the Adam optimizer
    """
    net = VAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss {}'.format(epoch, loss))

    return net