import pickle
import numpy as np


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.clfs = [None for _ in range(n_weakers_limit)]
        self.alpha = [0 for _ in range(n_weakers_limit)]


    def is_good_enough(self):
        '''Optional'''
        tp = 0
        tn = 0
        fp = 0
        fn = 0


    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples  # initialize with uniform distribution
        for i in range(self.n_weakers_limit):
            clf = self.weak_classifier(max_depth=2, random_state=2018)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X).reshape((-1, 1))
            error = sum((y_pred != y).flatten() * w)
            acc = (y_pred == y).sum() / float(y.shape[0])
            print('error', error)
            print('acc', acc)
            alpha = 0.5 * np.log(1/error - 1) if error <= 0.5 else 0
            if alpha == 0:
                break
            w = w * np.exp(-1 * alpha * y * y_pred).flatten()    # update w
            w = w / w.sum()  # renormalize w

            self.clfs[i] = clf
            self.alpha[i] = alpha


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pred = np.zeros(X.shape[0])
        for i in range(len(self.clfs)):
            pred += self.alpha[i] * self.clfs[i].predict(X)
        return pred.reshape((-1, 1))



    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pred = self.predict_scores(X)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = -1
        return pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
