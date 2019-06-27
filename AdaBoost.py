import random
from functools import partial

from keras.datasets import mnist
from sklearn.base import ClassifierMixin, BaseEstimator, clone
import numpy as np
from joblib import Parallel, delayed
from progressbar import SimpleProgress, Bar, Timer, Percentage, DynamicMessage, ProgressBar, AdaptiveETA

from sklearn.metrics import accuracy_score

def entropy(x):
    return (x * np.log(np.clip(x, np.finfo(x.dtype).eps, None))).sum()


def _parallel_build_stump(X, Y, sample_weights, labels_count, alpha, idx_feature):
    assert (np.isclose(sample_weights.sum(), 1))

    sorted_mask = X[:, idx_feature].argsort()

    sample_weights = sample_weights[sorted_mask]
    Y = Y[sorted_mask]
    X = X[sorted_mask, idx_feature]

    sum_left = np.zeros(labels_count)
    sum_right = np.zeros(labels_count)

    nr_instances = len(X)
    for higher_idx in range(nr_instances):
        sum_right[Y[higher_idx]] += sample_weights[higher_idx]

    if alpha == 0:
        prob_left = np.random.sample(labels_count)
        prob_left /= prob_left.sum()
    else:
        prob_left = np.zeros(labels_count)
        prob_left[np.random.choice(labels_count)] = 1

    prob_right = sum_right / sum_right.sum()

    label_right = sum_right.argmax()
    min = sum_right.sum() - sum_right[label_right] + alpha * entropy(prob_right)
    beststump = [X[0], (prob_left.argmax(), label_right), min, idx_feature, (prob_left, prob_right)]

    for idx_instance in range(1, nr_instances):
        treshold = X[idx_instance - 1] + (X[idx_instance] - X[idx_instance - 1]) / 2

        sum_right[Y[idx_instance - 1]] -= sample_weights[idx_instance - 1]
        sum_left[Y[idx_instance - 1]] += sample_weights[idx_instance - 1]

        prob_left = sum_left / sum_left.sum()
        prob_right = sum_right / sum_right.sum()

        label_left = sum_left.argmax()
        label_right = sum_right.argmax()

        entropy_left = entropy(sum_left)
        entropy_right = entropy(sum_right)

        weighted_sum = sum_left.sum() + sum_right.sum()

        entropy_mean = sum_left.sum() / weighted_sum * entropy_left + \
                  sum_right.sum() / weighted_sum * entropy_right

        errSum = 1 - (sum_left[label_left] + sum_right[label_right]) + alpha * entropy_mean

        if errSum < min:
            min = errSum
            beststump = [treshold, (label_left, label_right), errSum, idx_feature, (prob_left, prob_right)]

    return beststump


class DecisionStump(ClassifierMixin, BaseEstimator):
    def __init__(self, random_features=None, alpha=0.1, verbose=False, n_jobs=-1):
        self.verbose = verbose
        self.random_features = random_features
        self.stump = []
        self.idx_feature = -1
        self.labels_count = -1
        self.sample_weights = None
        self.n_jobs = n_jobs
        self.alpha = alpha

    def fit(self, X, Y, sample_weights=None):
        self.sample_weights = sample_weights
        self.labels_count = len(np.unique(Y))
        stump, idx_feature = self._build_stumps_parallelized(X, Y, self.random_features)
        self.stump = stump
        self.idx_feature = idx_feature

    def predict(self, X):
        result = self.decision_function(X)
        return result.argmax(axis=1)

    def decision_function(self, X):
        result = np.zeros((X.shape[0], self.labels_count))

        result.fill(-1 / (self.labels_count - 1))

        mask = X[:, self.idx_feature] < self.stump[0]
        result[mask, self.stump[1][0]] = 1
        result[~mask, self.stump[1][1]] = 1
        return result

    def predict_proba(self, X):
        result = np.zeros((X.shape[0], self.labels_count))

        mask = X[:, self.idx_feature] < self.stump[0]
        result[mask, :] = self.prob_left
        result[~mask, :] = self.prob_right

        return result

    def _build_stumps_parallelized(self, X, Y, random_features):
        """
        label (1,0) means x<T = 1, x>=T = 0
        label (0,1) means x<T = 0, x>=T = 1
        """
        nr_features = len(X[0])
        self.labels_count = len(np.unique(Y))

        if random_features is not None:
            stumps = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_parallel_build_stump, X, Y, self.sample_weights, self.labels_count, self.alpha))(idx_feature) for
                idx_feature in random.sample(range(0, nr_features), random_features))
        else:
            stumps = Parallel(n_jobs=self.n_jobs)(
                delayed(partial(_parallel_build_stump, X, Y, self.sample_weights, self.labels_count, self.alpha))(idx_feature) for
                idx_feature in range(0, nr_features))

        min = np.inf

        beststump_per_feature = []
        idx_feature = -1

        for stump in stumps:
            # optimizat, cautare dupa indexi, si scris de abia cand gaseste -todo
            if stump[2] < min:
                min = stump[2]
                beststump_per_feature = stump
                idx_feature = stump[3]

        self.prob_left = beststump_per_feature[4][0]
        self.prob_right = beststump_per_feature[4][1]

        return beststump_per_feature, idx_feature


def one_hot_encode_samme(array, n_labels):
    one_hot_encoded_array = np.zeros((array.shape[0], n_labels))
    one_hot_encoded_array.fill(-1 / (n_labels - 1))
    for idx_label in range(0, len(array)):
        one_hot_encoded_array[idx_label, array[idx_label]] = 1
    return one_hot_encoded_array


class AdaBoost(ClassifierMixin, BaseEstimator):

    def __init__(self, n_estimators=1, mode='SAMME', verbose=False,
                 base_estimator=DecisionStump(random_features=10)):
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.alphas = []
        self.stumps = []
        self.mode = mode
        self.n_labels = -1

    def fit(self, X, Y):
        """
        :param X: features
        :param Y: labels (+1 or -1)
        """
        # weighturile la cele gresite trebuie sa imi dea 1/2

        if self.verbose:
            widgets = [
                Percentage(),
                ' ', SimpleProgress(format="(%s)" % SimpleProgress.DEFAULT_FORMAT),
                ' ', Bar(),
                ' ', Timer(),
                ' ', AdaptiveETA(),
                ' ', DynamicMessage('training_acc'),
                ' ', DynamicMessage('weighted_err')
            ]

            bar = ProgressBar(max_value=self.n_estimators, widgets=widgets, redirect_stdout=True)
            bar.start()

        self.n_labels = len(np.unique(Y))
        random_chance_error = (self.n_labels - 1) / self.n_labels

        y_extended = one_hot_encode_samme(Y, self.n_labels)
        sample_weights = np.full((X.shape[0]), 1 / X.shape[0])

        training_decision_function = np.zeros((X.shape[0], self.n_labels))

        idx_iteration = 0
        retry_count = 0
        while idx_iteration < self.n_estimators:
            ds = clone(self.base_estimator)
            ds.fit(X, Y, sample_weights)
            if self.mode == 'SAMME':
                ds_prediction = ds.predict(X)
                ds_decision = one_hot_encode_samme(ds_prediction, self.n_labels)
            else:
                ds_proba = ds.predict_proba(X)
                np.clip(ds_proba, np.finfo(ds_proba.dtype).eps, None, out=ds_proba)
                ds_proba = np.log(ds_proba)

                ds_decision = (self.n_labels - 1) * (
                        ds_proba - (1 / self.n_labels) * ds_proba.sum(axis=1)[:, np.newaxis])
                ds_prediction = ds_decision.argmax(axis=1)

            err = sample_weights[Y != ds_prediction].sum()

            if err == 0:
                self.alphas.append(1)
                self.stumps.append(ds)
                break
            elif err >= random_chance_error:
                retry_count += 1
                if retry_count == 10:
                    print('no stump that could split the data, good enough, found')
                    break
                continue
            else:
                retry_count = 0
                if self.mode == 'SAMME':
                    self.alphas.append(np.log(1 - err) - np.log(err) + np.log(self.n_labels - 1))
                elif self.mode == 'SAMME.R':
                    self.alphas.append(1)

                assert (self.alphas[-1] > 0)

                training_decision_function += self.alphas[-1] * ds_decision
                training_prediction = training_decision_function.argmax(axis=1)
                acc_score = accuracy_score(Y, training_prediction)

                if self.verbose:
                    bar.update(idx_iteration, weighted_err=err, training_acc=acc_score)

                self.stumps.append(ds)

                if self.mode == 'SAMME':
                    weight_sum = 0

                    for idx_pred in range(0, len(Y)):
                        if Y[idx_pred] != ds_prediction[idx_pred]:
                            sample_weights[idx_pred] *= np.exp(self.alphas[-1])
                        weight_sum += sample_weights[idx_pred]

                    sample_weights /= weight_sum
                elif self.mode == 'SAMME.R':
                    op = -(self.n_labels - 1) / self.n_labels * y_extended * ds_proba
                    sample_weights = sample_weights * np.exp(op.sum(axis=1))
                    sample_weights /= sample_weights.sum()

            idx_iteration += 1
        if self.verbose:
            bar.update()
            bar.finish(dirty=True)

    def predict(self, X):
        result = self.decision_function(X)
        return result.argmax(axis=1)

    def predict_proba(self, X):
        prediction = self.decision_function(X)
        prediction /= self.n_labels
        prediction = np.exp((1. / (self.n_labels - 1)) * prediction)

        normalizer = prediction.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        prediction /= normalizer
        return prediction

    def decision_function(self, X):
        result = np.zeros((X.shape[0], self.n_labels))

        for alpha, stump in zip(self.alphas, self.stumps):
            if self.mode == 'SAMME':
                prediction = stump.predict(X)
                result += alpha * one_hot_encode_samme(prediction, self.K)
            elif self.mode == 'SAMME.R':
                proba = stump.predict_proba(X)
                np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
                ds_proba = np.log(proba)

                ds_decision = (self.n_labels - 1) * (
                        ds_proba - (1 / self.n_labels) * ds_proba.sum(axis=1)[:, np.newaxis])
                result += alpha * ds_decision
        return result


def oneVsRestSplit(X, Y):
    labels = np.unique(Y)
    for label in labels:
        yield label, X, (Y == label).astype(int)


class OneVsRest(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifiers = {}
        self.labels = []

    def fit(self, X, Y):
        for label, X, Y in oneVsRestSplit(X, Y):
            classifier = clone(self.classifier)
            classifier.fit(X, Y)

            self.classifiers[label] = classifier
            self.labels.append(label)

    def decision_function(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))

        for idx, label in enumerate(self.labels):
            classifier = self.classifiers[label]

            predictions[:, idx] = classifier.decision_function(X)[:, 1]

        return predictions

    def predict(self, X):
        results = self.decision_function(X)

        return np.vectorize(lambda x: self.labels[x])(results.argmax(axis=1))


########################################################################################################################

# X = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16],
#     [17, 18, 32, 20],
#     [21, 82, 23, 2],
#     [25, 26, 27, 28],
#     [23, 30, 345, 32],
#     [33, 34, 35, 36]
# ])
#
# Y = np.array([
#     0,
#     0,
#     0,
#     1,
#     2,
#     1,
#     2,
#     2,
#     3
# ])
#
# ab = AdaBoost(n_estimators=6, mode='SAMME.R', verbose=True, base_estimator=DecisionStump(random_features=None))
# ab.fit(X,Y)
# print(ab.predict(X))

########################################################################################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data('/home/augt/Downloads/mnist.npz')
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# ds = DecisionStump()
# ds.fit(X, Y, np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))

ab = AdaBoost(n_estimators=200, mode='SAMME.R', verbose=True, base_estimator=DecisionStump(random_features=10, alpha=0.00001))
ab.fit(x_train, y_train)
mnist_pred = ab.predict(x_test)
print('prediciton :=', mnist_pred)

print('acc: ', accuracy_score(y_test, mnist_pred))

########################################################################################################################

# iris = datasets.load_iris()
# X = iris.data  # we only take the first two features.
# y = iris.target
# from sklearn.model_selection import train_test_split
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# print(iris.data)
# print(iris.target)
#
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
#
# ab = AdaBoost(n_estimators=10, mode='SAMME')
# ab.fit(x_train, y_train)
# iris_pred = ab.predict(x_test)
# print('prediciton :=', iris_pred)
#
# print('acc: ', accuracy_score(y_test, iris_pred))
