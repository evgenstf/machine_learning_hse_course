import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, ElasticNet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import re
from sklearn.calibration import CalibratedClassifierCV 

import scipy.sparse
import numpy as np

train = open('../input/x_train.txt').read().split('\n')
train = train[:3600000]
test = open('../input/x_test.txt').read().split('\n')
test = test[:400000]
target = pd.read_csv('../input/y_train.csv')
target = np.array(target['Probability'].values)
magic = pd.read_csv('../input/magic.csv')
magic = np.array(magic['Probability'].values)
predicted = pd.read_csv('answer.csv')
predicted = np.array(predicted['Probability'].values)

print("score:", roc_auc_score(magic[:len(predicted)], predicted))
