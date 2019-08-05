import pystan
import numpy as np
import scipy.io
import os
import pickle

import time
import statistics as stat

# load features and labels

dir = <feature_and_label_file>                # '/mnt/4tb_other/satyajit/PycharmProjects/LSTM_Opportunity/Nested_CV/Features_and_labels/Outer/Set1'

mat = scipy.io.loadmat(dir) 

training_features = mat['training_features']
test_features = mat['test_features']
training_labels = mat['training_labels']
test_labels = mat['test_labels']
training_n_steps = mat['training_n_steps']
test_n_steps = mat['test_n_steps']


### prepare data for pystan

n_trainvid = training_n_steps.shape[0]
n_testvid = test_n_steps.shape[0]
n_class = 2


n_feature = 145

n_layers = 2    # model hyper-parameter, test 1,2,3
n_state = 3     # model hyper-parameter, test 1,2,3,4,5,6

# n_layers and n_state can be increased more depending on system limitations,
# but it is expected to observe decline in validation performance after a point due to overfitting.


num1 = n_state ** n_layers
num2 = (n_class * n_state) ** n_layers
num4 = (n_class * n_state) ** (n_layers - 1)

max_step = 1522

n_step = training_n_steps.ravel()  # reshape
test_step_length = test_n_steps.ravel()

X_train = training_features

X_test = test_features

y = training_labels

reg_l2_transition = 10
reg_l2_feature = 0.1


### jot all required data in a dictionary to pass to stan model
dat = dict(n_trainvid=n_trainvid, n_testvid=n_testvid, n_layers=n_layers, num1=num1, num2=num2,
               num4=num4, n_feature=n_feature, n_state=n_state, n_class=n_class, n_step=n_step,
                test_step_length=test_step_length, max_step=max_step, X_train=X_train,
                X_test=X_test, y=y, reg_l2_transition=reg_l2_transition, reg_l2_feature=reg_l2_feature)


# load compiled model, To compile model run compile_model.py
sm = pickle.load(open('<path_to_compiled_model>/FLDCRF_opportunity_HL.pkl', 'rb'))     
 

# train model and generate inferred quantities

fit = sm.optimizing(data=dat, algorithm='BFGS')


# save output as necessary, here we save the predicted class probability values [n_testvid, max_step, n_class]

HL = fit['Actual_proby']  

filename_new = ' '.join(('<path_to_save_file>', '.mat'))

scipy.io.savemat(filename_new.replace(" ", ""), {'HL': HL})
