from tqdm import tqdm

from src.engine import final_loss
from src.metrics.classification import accuracy, recall, precision, tpr, fpr


def classification_test(net, testloader, testset, device, criterion, thresholds):
    res = []
    for i, data in tqdm(enumerate(testloader), total=len(testset)):
        data = data['image']
        data = data.to(device)
        reconstruction, mu, logvar = net(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()

        res.append({'loss': loss, 'realLabel': data['label']})

    for t in thresholds:
        classify(res, t)


def classify(res, threshold):
    data = []
    for i, r in res:
        label = 0 if r['loss'] < threshold else 1
        data.append({'label': label, 'realLabel': data['label']})

    acc = accuracy(data)
    pre = precision(data)
    rec = recall(data)
    tp_rate = tpr(data)
    fp_rate = fpr(data)

    print('-----------------------------')
    print('Threshold: ' + threshold)
    print('Accuracy: ' + acc)
    print('Precision: ' + pre)
    print('Recall: ' + rec)
    print('TPR: ' + tp_rate)
    print('FPR' + fp_rate)
