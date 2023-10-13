import numpy as np


class Perceptron:
    def __init__(self, train_set, test_set, epoch=50, learning_rate=0.0001):
        self.epoch = epoch
        self.train_set = train_set
        self.test_set = test_set
        self.learning_rate = learning_rate
        self.w = 0
        self.b = 0

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
                data.append([d / cur_max for d in cur_data])
                labels.append(1 if line[0] < 5 else -1)

        return np.mat(data), np.mat(labels).T

    def train(self):
        data, labels = self.load_data(self.train_set)
        data_count, data_length = data.shape
        w = np.zeros((1, data_length))
        b = 0

        print("training...")
        for e in range(self.epoch):
            loss = 0
            for i in range(data_count):
                x = data[i]
                y = labels[i]
                result = -1 * y * (w * x.T + b)
                if result >= 0:
                    w += y * x * self.learning_rate
                    b += y * self.learning_rate
                    loss += result.item(0, 0)

            print("epoch: {0}  loss: {1}".format(e, loss))

        self.w = w
        self.b = b

    def test(self):
        data, labels = self.load_data(self.test_set)
        data_count = data.shape[0]
        w = self.w
        b = self.b

        print("testing...")
        wrong_count = 0
        for i in range(data_count):
            x = data[i]
            y = labels[i]
            if -1 * y * (w * x.T + b) >= 0:
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data_count))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    perceptron = Perceptron(train_set, test_set)
    perceptron.train()
    perceptron.test()
