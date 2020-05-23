import pystan
import numpy as np
import scipy.io
import os
import pickle

# load('C:\Satyajit\OpportunityUCIDataset\OpportunityUCIDataset\dataset\Our_data\Evaluation\Nested_CV\S2\Features_and_labels\Outer\Set3', 'training_features', 'test_features', 'training_labels', 'test_labels', 'training_n_steps', 'test_n_steps');

dir = '/mnt/4tb_other/satyajit/PycharmProjects/FLDCRF_Opportunity/S2/Features_and_labels/Outer/'

os.chdir(dir)

for feature_file in sorted(os.listdir(dir)):
    print(os.path.abspath(feature_file))

    mat = scipy.io.loadmat(os.path.abspath(feature_file))  # /mnt/4tb_other/satyajit/PycharmProjects/LSTM_Opportunity/Nested_CV/Features_and_labels/Inner/Set1/Set12.mat

    training_features = mat['training_features']         # must be [n_trainvid, max_step, n_feature]
    test_features = mat['test_features']                 # must be [n_testvid, max_step, n_feature]
    training_labels = mat['training_labels']             # must be [n_trainvid, max_step, n_layers], n_layers equal to number of label categories in this code
    test_labels = mat['test_labels']                     # must be [n_testvid, max_step, n_layers], n_layers equal to number of label categories in this code
    training_n_steps = mat['training_n_steps']           # must be [n_trainvid]
    test_n_steps = mat['test_n_steps']                   # must be [n_testvid]

    n_trainvid = training_n_steps.shape[0]
    n_testvid = test_n_steps.shape[0]
    n_layers = 2

    n_feature_cont = 145
    n_state = np.array([3, 4])
    n_label = np.array([4, 2])

    num1 = n_state[0] * n_state[1]                                      # model component, do not change
    num2 = (n_label[0] * n_state[0]) * (n_label[1] * n_state[1])        # model component, do not change

    print(num1)
    print(num2)

    max_step = 1522

    n_step = training_n_steps.ravel()  # reshape
    test_step_length = test_n_steps.ravel()

    X_train_cont = training_features
    X_test_cont = test_features

    y = training_labels

    reg_l2_transition = 10
    reg_l2_feature = 0.1

    dat = dict(n_trainvid=n_trainvid, n_testvid=n_testvid, n_layers=n_layers, num1=num1, num2=num2,
                n_feature_cont=n_feature_cont, n_state=n_state, n_label=n_label, n_step=n_step,
                test_step_length=test_step_length, max_step=max_step, X_train_cont=X_train_cont,
                X_test_cont=X_test_cont, y=y, reg_l2_transition=reg_l2_transition, reg_l2_feature=reg_l2_feature)

    # load compiled model, To compile model run compile_model.py
    sm = pickle.load(open('/mnt/4tb_other/satyajit/PycharmProjects/FLDCRF_Opportunity/Stan_codes/FLDCRF_multilabel.pkl', 'rb'))

    # train model and generate inferred quantities
    fit = sm.optimizing(data=dat, algorithm='BFGS')

    # save output as necessary, here we save the predicted class probability values [n_testvid, max_step, n_class] for each label category
    locomotion = fit['Actual_proby1']
    HL = fit['Actual_proby2']

    filename_new = ' '.join(('/mnt/4tb_other/satyajit/PycharmProjects/FLDCRF_Opportunity/S2/Results/Outer/FLDCRF_cross_links_pystan/FLDCRF_34/', feature_file, '.mat'))

    scipy.io.savemat(filename_new.replace(" ", ""), {'locomotion': locomotion, 'HL': HL})
