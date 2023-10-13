import numpy as np


class NaiveBayes(object):
    def __init__(self, train_set, test_set, label_num=10, set_num=2, _lambda=1):
        self.train_set = train_set
        self.test_set = test_set
        self._lambda = _lambda
        self.label_num = label_num
        self.set_num = set_num
        self.labels_prob = []
        self.conditional_prob = []

    def load_data(self, dataset):
        print("Loading data from    {}".format(dataset))

        data = []
        labels = []
        with open(dataset, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(",")
                line = [int(l) for l in line]

                cur_data = line[1:]
                cur_max = max(cur_data)
                data.append([int(d > (cur_max / 2)) for d in cur_data])
                # data.append(line[1:])
                labels.append(line[0])

        return np.mat(data), np.mat(labels).T

    def calculate_probabilities(self, data, labels):
        labels_prob = [0] * self.label_num
        for label in labels:
            labels_prob[int(label)] += 1

        conditional_prob = np.zeros((self.label_num, self.set_num, data.shape[1])) + self._lambda
        for index1, datum in enumerate(data):
            label = int(labels[index1])
            datum = datum.tolist()[0]
            for index2, dt in enumerate(datum):
                conditional_prob[label][dt][index2] += 1

        for index, count in enumerate(labels_prob):
            conditional_prob[index] /= count + self.set_num * self._lambda
        labels_prob = (np.array(labels_prob) + self._lambda) / (labels.shape[0] + len(labels_prob) * self._lambda)

        return labels_prob, conditional_prob

    def calculate_Pxy(self, label, data):
        Pxy = 1
        for index, datum in enumerate(data):
            Pxy *= self.conditional_prob[label][datum][index]

        return Pxy

    def train(self):
        data, labels = self.load_data(self.train_set)
        print("Training to get probabilities")
        self.labels_prob, self.conditional_prob = self.calculate_probabilities(data, labels)

    def test(self):
        data, labels = self.load_data(self.test_set)
        wrong_count = 0

        print("testing...")
        for index, real_label in enumerate(labels):
            test_prob = []
            real_label = int(real_label)
            for label, Py in enumerate(self.labels_prob):
                Pxy = self.calculate_Pxy(label, data[index].tolist()[0])
                test_prob.append(Py * Pxy)
            if test_prob.index(max(test_prob)) != real_label:
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data.shape[0]))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    naive_bayes = NaiveBayes(train_set, test_set)
    naive_bayes.train()
    naive_bayes.test()
