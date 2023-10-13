import numpy as np


class DecisionTree(object):
    def __init__(self, train_set, test_set, label_num=10, epsilon=0.1):
        self.train_set = train_set
        self.test_set = test_set
        self.epsilon = epsilon
        self.label_num = label_num
        self.tree = None
        self.level = 0

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

    def get_major_label(self, labels):
        label_count = [0] * self.label_num
        for label in labels:
            label_count[int(label)] += 1

        return label_count.index(max(label_count))

    def calculate_entropy(self, labels):
        entropy = [0] * self.label_num
        for label in labels:
            entropy[int(label)] += 1
        entropy = np.array(entropy) / len(labels)
        entropy = np.nan_to_num(entropy * np.log2(entropy))
        entropy = -np.sum(entropy)

        return entropy

    def get_part_labels(self, figure, labels):
        part_labels = [[], []]
        figure_frequency = [0, 0]
        for index, fg in enumerate(figure):
            fg = int(fg)
            part_labels[fg].append(int(labels[index]))
            figure_frequency[fg] += 1
        figure_frequency = np.array(figure_frequency) / figure.shape[0]

        return part_labels, figure_frequency

    def get_max_gain(self, data, labels):
        # 经验熵
        entropy = self.calculate_entropy(labels)

        # 条件熵
        gains = []
        for i in range(data.shape[1]):  # 需优化
            cur_figure = data[:, i]
            part_labels, figure_frequency = self.get_part_labels(cur_figure, labels)
            gain = entropy - figure_frequency[0] * self.calculate_entropy(part_labels[0]) - figure_frequency[
                1] * self.calculate_entropy(part_labels[1])
            gains.append(gain)

        max_gain = max(gains)
        Ag = gains.index(max_gain)

        return Ag, max_gain

    def part_dataset(self, figure, data, labels):
        part_labels, _ = self.get_part_labels(figure, labels)
        part_data = [[], []]
        for index, fg in enumerate(figure):
            datum = data[index].tolist()[0]
            part_data[int(fg)].append(datum)

        return part_data, part_labels

    def create_tree(self, data, labels):
        self.level += 1
        print("Creating for level {}".format(self.level))

        label_set = {int(label) for label in labels}
        if len(label_set) == 1:
            return label_set.pop()
        if data.shape[1] == 0:
            return self.get_major_label(labels)

        Ag, max_gain = self.get_max_gain(data, labels)
        if max_gain < self.epsilon:
            return self.get_major_label(labels)

        tree = {Ag: {}}
        part_data, part_labels = self.part_dataset(data[:, Ag], np.delete(data, Ag, axis=1), labels)
        tree[Ag][0] = self.create_tree(np.mat(part_data[0]), np.mat(part_labels[0]).T)
        self.level -= 1
        tree[Ag][1] = self.create_tree(np.mat(part_data[1]), np.mat(part_labels[1]).T)

        return tree

    def predict(self, tree, data):
        if type(tree) == int:
            return tree

        result = -1
        for node in tree.keys():
            result = self.predict(tree[node][int(data.item(0, node))], data)

        return result

    def train(self):
        data, labels = self.load_data(self.train_set)
        print("Training to create decision tree")
        self.level = 0
        self.tree = self.create_tree(data, labels)

    def test(self):
        data, labels = self.load_data(self.test_set)
        wrong_count = 0

        print("Testing...")
        for index, datum in enumerate(data):
            test_label = self.predict(self.tree, datum)

            if test_label != int(labels[index]):
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data.shape[0]))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    decision_tree = DecisionTree(train_set, test_set)
    decision_tree.train()
    decision_tree.test()
