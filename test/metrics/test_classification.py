from unittest import TestCase

from metrics.classification import accuracy, get_metrics, precision, recall, tpr, fpr


class Test(TestCase):

    @staticmethod
    def create_data(tp, tn, fp, fn):
        data = []

        i = 0
        for j in range(tp):
            data.append({'idx': i, 'label': 0, 'realLabel': 0})
            i += 1
        for j in range(tn):
            data.append({'idx': i, 'label': 1, 'realLabel': 1})
            i += 1
        for j in range(fp):
            data.append({'idx': i, 'label': 0, 'realLabel': 1})
            i += 1
        for j in range(fn):
            data.append({'idx': i, 'label': 1, 'realLabel': 0})
            i += 1

        return data

    def test_accuracy(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = accuracy(data)

        self.assertAlmostEquals(res, 0.91, places=2, msg='Accuracy test failed')

    def test_precision(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = precision(data)

        self.assertAlmostEquals(res, 0.5, places=2, msg='Precision test failed')

    def test_recall(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = recall(data)

        self.assertAlmostEquals(res, 0.11, places=2, msg='Recall test failed')

    def test_tpr(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = tpr(data)

        self.assertAlmostEquals(res, 0.11, places=2, msg='Recall test failed')

    def test_fpr(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = fpr(data)

        self.assertAlmostEquals(res, 0.01, places=2, msg='Recall test failed')

    def test_get_metrics(self):
        # 0 HEALTHY
        # 1 DISEASE

        TP = 1
        TN = 90
        FP = 1
        FN = 8

        data = Test.create_data(TP, TN, FP, FN)

        res = get_metrics(data)

        self.assertEqual(res['TP'], TP)
        self.assertEqual(res['TN'], TN)
        self.assertEqual(res['FP'], FP)
        self.assertEqual(res['FN'], FN)
