from feature import NPDFeature
import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

N_FACES = 500
GRAY_SIZE = 24

# Hyper-params
TEST_SIZE = 0.2
N_WEAKER_LIMITS = 15

FEAT_FILE_PATH = 'feat.dat'
IMAGE_PATH = './datasets/original'


def train_val_split(X, y, test_size, shuffle=False):
    nrw_train = int(2 * N_FACES * (1 - test_size))
    return X[:nrw_train], y[:nrw_train], X[nrw_train:], y[nrw_train:]


def load_feats_data():
    if os.path.exists(FEAT_FILE_PATH):
        return pickle.load(open(FEAT_FILE_PATH, 'rb'))
    return create_npd_features()


def create_npd_features():
    grays = []
    lbls = []
    print('Converting images to grayscale with size ({0}, {0})...'.format(GRAY_SIZE))
    for i in range(N_FACES):
        for cla in ['face', 'nonface']:
            label = 1 if cla == 'face' else -1
            img = cv2.imread('{0}/{1}/{1}_{2:0>3d}.jpg'.format(IMAGE_PATH, cla, i))
            plt.figure()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (GRAY_SIZE, GRAY_SIZE))
            gray = (gray * 255).astype(np.uint8)    # convert float64 to 0~255 integer

            grays.append(gray)
            lbls.append(label)

    feats = []
    print('Extracting features...')
    for i in range(len(grays)):
        feat = NPDFeature(grays[i]).extract()
        feats.append(feat)
        print('Image {}...'.format(i))

    feats = np.array(feats)
    lbls = np.array(lbls).reshape((-1, 1))
    # print(feats.shape, lbls.shape)
    pickle.dump((feats, lbls), open(FEAT_FILE_PATH, 'wb'))
    return feats, lbls

#
# def predict_accuracy(y_true, y_pred):
#     return (y_true == y_pred).sum() / y_true.shape[0]


if __name__ == "__main__":
    print('Dataset loading...')
    X, y = load_feats_data()
    print(X.shape)
    X_train, y_train, X_val, y_val = train_val_split(X, y, test_size=TEST_SIZE)

    #  Adaboost
    adaBoost_clf = AdaBoostClassifier(DecisionTreeClassifier, N_WEAKER_LIMITS)
    print('AdaBoost fitting...')
    t = time.time()
    adaBoost_clf.fit(X_train, y_train)
    print('Finished. Time spent: {}s'.format(int(time.time() - t)))
    y_pred = adaBoost_clf.predict(X_val)
    # print(predict_accuracy(y_val, y_pred))

    with open('./classifier_report/classifier_report.txt', 'w+') as f:
        f.write(classification_report(y_val, y_pred, target_names=['positive', 'negative'], digits=3))

    # simple DecisionTreeClassifier
    print('DecisionTreeClassifier fitting...')
    n_samples = X_train.shape[0]
    clf = DecisionTreeClassifier(random_state=2018, max_depth=2)
    w = np.ones(n_samples) / n_samples
    t = time.time()
    clf.fit(X_train, y_train, sample_weight=w)
    print('Finished. Time spent: {}s'.format(int(time.time() - t)))
    y_pred = clf.predict(X_val).reshape(-1, 1)

    with open('./classifier_report/dt_classifier_report.txt', 'w+') as f:
        f.write(classification_report(y_val, y_pred, target_names=['positive', 'negative'], digits=3))

