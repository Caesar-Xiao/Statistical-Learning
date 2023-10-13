import numpy as np


class Logistic:
    def __init__(self, train_set, test_set, epoch=100, learning_rate=0.0001):
        self.epoch = epoch
        self.train_set = train_set
        self.test_set = test_set
        self.learning_rate = learning_rate
        self.w = 0

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
                data.append([d / cur_max for d in cur_data] + [1])
                labels.append(1 if line[0] == 0 else 0)
                # labels.append(1 if line[0] < 5 else 0)

        return np.array(data), np.array(labels)

    def train(self):
        data, labels = self.load_data(self.train_set)
        data_count, data_length = data.shape
        w = np.zeros(data_length)

        print("training...")
        for e in range(self.epoch):
            for i in range(data_count):
                x = data[i]
                y = labels[i]
                wx = np.dot(w, x)
                w += self.learning_rate * (y * x - (x * np.exp(wx)) / (1 + np.exp(wx)))

            if e % 10 == 0:
                print("epoch: {0}".format(e))

        self.w = w

    def test(self):
        data, labels = self.load_data(self.test_set)
        data_count = data.shape[0]
        w = self.w

        print("testing...")
        wrong_count = 0
        for i in range(data_count):
            x = data[i]
            y = labels[i]
            wx = np.dot(w, x)
            p = np.exp(wx) / (1 + np.exp(wx))
            test_label = 1 if p >= 0.5 else 0
            if test_label != y:
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data_count))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    perceptron = Logistic(train_set, test_set)
    perceptron.train()
    perceptron.test()
