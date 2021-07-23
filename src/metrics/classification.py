def accuracy(data):
    metrics = get_metrics(data)

    TP = metrics['TP']
    TN = metrics['TN']
    FP = metrics['FP']
    FN = metrics['FN']

    return (TP + TN) / (TP + TN + FP + FN)


def precision(data):
    metrics = get_metrics(data)

    TP = metrics['TP']
    FP = metrics['FP']

    if TP + FP == 0:
        return 0

    return TP / (TP + FP)


def recall(data):
    metrics = get_metrics(data)

    TP = metrics['TP']
    FN = metrics['FN']

    if TP + FN == 0:
        return 0

    return TP / (TP + FN)


def tpr(data):
    return recall(data)


def fpr(data):
    metrics = get_metrics(data)

    FP = metrics['FP']
    TN = metrics['TN']

    if FP + TN == 0:
        return 0

    return FP / (FP + TN)


def get_metrics(data):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for d in data:
        if d['label'] == 0 and d['realLabel'] == 0:
            TP += 1
        elif d['label'] == 1 and d['realLabel'] == 1:
            TN += 1
        elif d['label'] == 1 and d['realLabel'] == 0:
            FN += 1
        else:
            FP += 1

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}