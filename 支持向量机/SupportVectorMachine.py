import numpy as np


class SVM(object):
    def __init__(self, train_set, test_set, epochs=100, C=200, sigma=10, ksi=0.001, threshold=0.0001):
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs

        self.C = C
        self.sigma = sigma
        self.ksi = ksi
        self.threshold = threshold

        self.b = 0
        self.train_data, self.train_labels = self.load_data(self.train_set)
        self.train_data, self.train_labels = self.train_data[:1000], self.train_labels[:1000]
        self.alpha = np.array([0] * self.train_data.shape[0], float)
        self.support_vectors = []

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
                labels.append(1 if line[0] == 0 else -1)
                # labels.append(1 if line[0] < 5 else 0)

        return np.array(data), np.array(labels)

    def kernel_function(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) / (2 * self.sigma ** 2))

    def calculate_gx(self, x):
        k = np.array([self.kernel_function(xi, x) for xi in self.train_data])
        temp = self.alpha * self.train_labels * k

        return np.sum(temp) + self.b

    def get_Ei(self, x, y):
        return self.calculate_gx(x) - y

    def validate_KKT(self, alpha, x, y):
        y_gx = y * self.calculate_gx(x)
        difference = abs(y_gx - 1)

        if abs(alpha - 0) < self.ksi and y_gx >= 1:
            return True, 0
        elif -self.ksi < alpha < self.C + self.ksi and difference < self.ksi:
            return True, 0
        elif abs(alpha - self.C) < self.ksi and y_gx <= 1:
            return True, 0

        return False, difference

    def get_unsatisfied_alpha(self):
        unsatisfied_alpha = []
        for index, alpha in enumerate(self.alpha):
            satisfied, difference = self.validate_KKT(alpha, self.train_data[index], self.train_labels[index])
            if not satisfied:
                unsatisfied_alpha.append(index)

        return unsatisfied_alpha

    def get_alpha1_alpha2(self, Ei, alpha1, unsatisfied_alpha):
        E1 = Ei[alpha1]
        unsatisfied_Ei = Ei[unsatisfied_alpha]

        alpha2 = np.argmax(np.abs(unsatisfied_Ei - E1))
        alpha2 = unsatisfied_alpha[alpha2]

        return alpha2

    def update_params(self, Ei, alpha1, alpha2):
        E1, E2 = Ei[alpha1], Ei[alpha2]
        x1, x2 = self.train_data[alpha1], self.train_data[alpha2]
        y1, y2 = self.train_labels[alpha1], self.train_labels[alpha2]
        alpha1_old, alpha2_old = self.alpha[alpha1], self.alpha[alpha2]
        k11, k12, k22 = self.kernel_function(x1, x1), self.kernel_function(x1, x2), self.kernel_function(x2, x2)

        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha2_old + alpha1_old - self.C)
            H = min(self.C, alpha2_old + alpha1_old)
        if L == H:
            return alpha1_old, alpha2_old, self.b

        # update alpha
        alpha2_new = alpha2_old + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
        if alpha2_new < L:
            alpha2_new = L
        elif alpha2_new > H:
            alpha2_new = H

        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

        # update b
        b1_new = -E1 - y1 * k11 * (alpha1_new - alpha1_old) - y2 * k12 * (alpha1_new - alpha2_old) + self.b
        b2_new = -E2 - y1 * k12 * (alpha1_new - alpha1_old) - y2 * k22 * (alpha1_new - alpha2_old) + self.b
        if 0 < alpha1_new < self.C:
            b_new = b1_new
        elif 0 < alpha2_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new) / 2

        return alpha1_new, alpha2_new, b_new

    def train(self):
        print("Presetting...")
        Ei = np.array([self.get_Ei(x, self.train_labels[i]) for i, x in enumerate(self.train_data)])

        print("Training...")
        hasOptimized = False
        for epoch in range(self.epochs):
            unsatisfied_alpha = self.get_unsatisfied_alpha()
            if epoch % 10 == 0:
                print("Epoch: {0}\tUnsatisfied Alpha: {1}".format(epoch, len(unsatisfied_alpha)))

            for alpha_i in unsatisfied_alpha:
                alpha_j = self.get_alpha1_alpha2(Ei, alpha_i, unsatisfied_alpha)

                alpha1_old, alpha2_old = self.alpha[alpha_i], self.alpha[alpha_j]
                alpha1_new, alpha2_new, b_new = self.update_params(Ei, alpha_i, alpha_j)
                if alpha1_old == alpha1_new and alpha2_old == alpha2_new and self.b == b_new:
                    continue

                self.alpha[alpha_i], self.alpha[alpha_j] = alpha1_new, alpha2_new
                self.b = b_new

                x1, x2 = self.train_data[alpha_i], self.train_data[alpha_j]
                y1, y2 = self.train_labels[alpha_i], self.train_labels[alpha_j]

                Ei[alpha_i] = self.get_Ei(x1, y1)
                Ei[alpha_j] = self.get_Ei(x2, y2)

                if abs(alpha1_old - alpha1_new) > self.threshold or abs(alpha2_old - alpha2_new) > self.threshold:
                    hasOptimized = True

            if not hasOptimized:
                break

        for index, alpha in enumerate(self.alpha):
            if alpha > 0:
                self.support_vectors.append(index)

        print("Support Vectors: {0} {1}\n".format(len(self.support_vectors), self.support_vectors))

    def test(self):
        data, labels = self.load_data(self.test_set)
        data, labels = data[:100], labels[:100]

        print("Testing...")
        wrong_count = 0
        for index, datum in enumerate(data):
            test_label = self.b
            for vector in self.support_vectors:
                test_label += self.alpha[vector] * self.train_labels[vector] * self.kernel_function(
                        self.train_data[vector], datum)

            test_label = np.sign(test_label)
            if test_label != labels[index]:
                wrong_count += 1

        print("Accuracy: {0}".format(1 - wrong_count / data.shape[0]))


if __name__ == '__main__':
    train_set = '../Mnist/mnist_train.csv'
    test_set = '../Mnist/mnist_test.csv'
    svm = SVM(train_set, test_set)
    svm.train()
    svm.test()
