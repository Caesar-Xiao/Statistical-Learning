import numpy as np


class MaxEntropy(object):
    def __init__(self, train_set, test_set, epochs=100, M=10000, label_count=2, feature_count=2):
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs
        self.M = M
        self.label_count = label_count
        self.feature_count = feature_count
        self.xy_counts = []
        self.xy_num = 0
        self.xy2id = []
        self.w = []

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
                half_cur_max = max(cur_data) / 2
                data.append([int(d > half_cur_max) for d in cur_data])
                labels.append(1 if line[0] == 0 else 0)
                # labels.append(1 if line[0] < 5 else 0)

        return np.array(data), np.array(labels)

    def calculate_Ep_hat(self, data, labels):
        xy_counts = [{} for i in range(data.shape[1])]
        for index1, datum in enumerate(data):
            for index2, feature in enumerate(datum):
                tuple_key = (feature, labels[index1])
                cur_dict = xy_counts[index2]
                if tuple_key not in cur_dict:
                    cur_dict[tuple_key] = 0
                cur_dict[tuple_key] += 1

        Ep_hat = []
        for index, counts in enumerate(xy_counts):
            temp_dict = {}
            for tuple_key in counts:
                temp_dict[tuple_key] = len(Ep_hat)
                Ep_hat.append(counts[tuple_key] / data.shape[0])
            self.xy2id.append(temp_dict)
        Ep_hat = np.array(Ep_hat)

        self.xy_num = Ep_hat.shape[0]
        self.xy_counts = xy_counts

        return Ep_hat

    def calculate_Ep(self, data):
        P_hat_x = [[0] * self.feature_count for i in range(data.shape[1])]
        for datum in data:
            for index, feature in enumerate(datum):
                P_hat_x[index][feature] += 1
        P_hat_x = np.array(P_hat_x) / data.shape[0]

        Ep = [0] * self.xy_num
        self.w = [0] * self.xy_num
        for datum in data:
            Py1x = self.calculate_Py1x(datum)
            for index, id_dict in enumerate(self.xy2id):
                for tuple_key, id in id_dict.items():
                    Ep[id] += P_hat_x[index][tuple_key[0]] * Py1x[tuple_key[1]]  # Py1x[tuple_key[1]] / data.shape[0]

        return np.array(Ep)

    def calculate_Py1x(self, X):
        Py1x = [0] * self.feature_count
        for y in range(self.feature_count):
            for index, x in enumerate(X):
                tuple_key = (x, y)
                if tuple_key in self.xy2id[index]:
                    id = self.xy2id[index][tuple_key]
                    Py1x[y] += self.w[id]

        Py1x = np.exp(np.array(Py1x))
        Py1x /= np.sum(Py1x)

        return Py1x

    def train(self):
        # w <- w + delta
        # delta = 1 / M * log( Ep_hat / Ep )
        data, labels = self.load_data(self.train_set)
        data = data[:1000]
        labels = labels[:1000]

        print("Preset Ep_hat")
        Ep_hat = self.calculate_Ep_hat(data, labels)

        print("Training...")
        for epoch in range(self.epochs):
            Ep = self.calculate_Ep(data)
            delta = np.log(Ep_hat / Ep) / self.M
            self.w += delta

            if epoch % 10 == 0:
                print("Epoch: {0}".format(epoch))
                print(delta)

    def test(self):
        data, labels = self.load_data(self.test_set)
        data = data[:500]
        labels = labels[:500]

        print("Testing...")
        wrong_count = 0
        for index, datum in enumerate(data):
            Py1x = self.calculate_Py1x(datum)
            test_label = Py1x.tolist().index(max(Py1x))
            if test_label != labels[index]:
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data.shape[0]))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    max_entropy = MaxEntropy(train_set, test_set)
    max_entropy.train()
    max_entropy.test()
