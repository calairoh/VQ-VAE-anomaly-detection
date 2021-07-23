from tqdm import tqdm

from src.engine import final_loss


def classification_test(net, testloader, testset, device, criterion, threshold):
    res = []
    for i, data in tqdm(enumerate(testloader), total=len(testset)):
        data = data['image']
        data = data.to(device)
        reconstruction, mu, logvar = net(data[None, ...])
        bce_loss = criterion(reconstruction, data[None, ...])
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()

        label = 0 if loss < threshold else 1

    res.append({'idx': i, 'label': label, 'realLabel': data['label']})



