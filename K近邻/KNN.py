import numpy as np


class KNN:
    def __init__(self, train_set, test_set, top_K=25):
        self.train_set = train_set
        self.test_set = test_set
        self.top_K = top_K

    def load_data(self, dataset):
        print("Loading data from    {}".format(dataset))

        data = []
        labels = []
        with open(dataset, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(",")
                line = [int(l) for l in line]

                data.append(line[1:])
                labels.append(line[0])

        return np.mat(data), np.mat(labels).T

    def get_distance(self, train, test):
        return np.sqrt(np.sum(np.square(train - test)))

    def get_k_set(self, train_data, test):
        distances = []
        for train in train_data:
            distances.append(self.get_distance(train, test))
        k_set = np.argsort(np.array(distances))[:self.top_K]

        return k_set

    def classify(self):
        train_data, train_labels = self.load_data(self.train_set)
        test_data, test_labels = self.load_data(self.test_set)

        wrong_count = 0
        test_data = test_data[:100]  # 取100条
        for index, test in enumerate(test_data):
            k_set = self.get_k_set(train_data, test)
            k_labels = [0] * 10
            for i in k_set:
                label = train_labels[i].item(0, 0)
                k_labels[label] += 1

            test_label = k_labels.index(max(k_labels))
            if test_label != test_labels[index].item(0, 0):
                wrong_count += 1

            if index % 10 == 0:
                print("Classifying... {0}%".format(round(index / test_data.shape[0] * 100, 2)))
        print("Accuracy: {0}".format(1 - wrong_count / test_data.shape[0]))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    knn = KNN(train_set, test_set)
    knn.classify()
