#!/usr/bin/env python
# -*- coding utf-8 -*-
# author:yuboona
# datetime:17-12-29 下午5:06
# software:PyCharm

from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import  train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt


a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float64)
print(preprocessing.scale(a))