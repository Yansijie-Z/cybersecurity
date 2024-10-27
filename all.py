import numpy as np
import pandas as pd
# a feature tranformer  , help to capture complex patterns
# approximates radial basis function
# map input features into a higher-dimensional space
from sklearn.kernel_approximation import RBFSampler
# large-scale dataset usable linear classifier
# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import(roc_curve, classification_report,auc)
from sklearn.preprocessing import Normalizer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


traindata1=pd.read_csv('KDDTrain+_20Percent.txt',header=None)
testdata1=pd.read_csv('KDDTest+.txt', header=None)
