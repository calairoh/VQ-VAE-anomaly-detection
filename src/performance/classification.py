from tqdm import tqdm

from src.engine import final_loss
from src.metrics.classification import accuracy, recall, precision, tpr, fpr


def classification_performance_computation(net, testloader, testset, device, criterion, thresholds):
    net.eval()

    res = []
    for i, data in tqdm(enumerate(testloader), total=len(testset)):
        img = data['image']
        img = img.to(device)
        reconstruction, mu, logvar = net(img)
        bce_loss = criterion(reconstruction, img)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()

        res.append({'loss': loss, 'realLabel': data['label']})

    for t in thresholds:
        classify(res, t)


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

    print('-----------------------------')
    print('Threshold: ' + str(threshold))
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(pre))
    print('Recall: ' + str(rec))
    print('TPR: ' + str(tp_rate))
    print('FPR' + str(fp_rate))
