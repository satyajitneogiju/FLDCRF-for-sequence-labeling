# FLDCRF-for-sequence-labeling
Factored Latent-Dynamic Conditional Random Fields for sequence labeling and prediction. FLDCRF outperforms LSTM on small/medium datasets (&lt;40,000 instances) across 6 experiments on 4 datasets.
This repository contains Factored Latent-Dynamic Conditional Random FIelds (FLDCRF) codes for single and multi-label sequence labeling/prediction. FLDCRF [1] is a single and multi-label generalization of LDCRF [2]. In our single-label experiments on 4 datasets (NTU [1], UCI opportunity [3], UCI gesture phase [4], JAAD [5]) across 6 experiments, FLDCRF improves over LDCRF performance and outperforms LSTM and LSTM-CRF on all test sets. All datasets are small/medium sized with < 40,000 instances. 
Additionally, LSTM is known for tedious hyper-parameter optimization process, big-data-driven performance and long training time. FLDCRF offers easier model selection (with no need to tune number of epochs), performs better than LSTM on small/medium datasets, is much easier to comprehend and takes significantly lesser training time, even without GPU implementation. We also find FLDCRF to generalize better over validation and test sets, i.e., selected LSTM models perform worse on test sets than selected FLDCRF models, even though in some cases LSTM models perform better on validation. Such inconsistency across validation and test, blurry intuition and tedious model selection makes LSTM hard for industrial applications.
In our recent multi-label experiment on UCI opportunity dataset [3], FLDCRF outperformed LDCRF, Factorial CRF, LSTM-CRF, LSTM and a LSTM multi label model. We will update the paper link for multi-label experiment shortly.
To run our codes, you need to install pystan - 
pip install pystan
PyStan is the Python interface of Stan probabilistic modeling language. For pystan documentation, refer to this - https://pystan.readthedocs.io/en/latest/getting_started.html. For stan language documentation, please check this - https://mc-stan.org/users/documentation/. Stan provides easy-to-use BFGS, L-BFGS and Newton optimization algorithms with effective stopping criteria. As we do not need to tune the number of training epochs for FLDCRF, Pystan is a very useful platform.
After having the codes in your system, you need to prepare the data as instructed in the python code (.py) and stan code (.stan). Then compile the stan model for the first time by running complie_model.py. Once the model is compiled, provide the path to the compiled  model in the python code - 
# load compiled model, To compile model run compile_model.py
sm = pickle.load(open('<path_to_compiled_model>/FLDCRF_opportunity_HL.pkl', 'rb'))    

Next, load the data and prepare a dictionary to pass the data to the stan model through the python model FLDCRF_pystan.py.

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
# n_layers and n_state can be increased more depending on system limitations, but it is expected to observe decline in validation performance after a point due to overfitting.

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
dat = dict(n_trainvid=n_trainvid, n_testvid=n_testvid, n_layers=n_layers, num1=num1, num2=num2, num4=num4, n_feature=n_feature, n_state=n_state, n_class=n_class, n_step=n_step,test_step_length=test_step_length, max_step=max_step, X_train=X_train, X_test=X_test, y=y, reg_l2_transition=reg_l2_transition, reg_l2_feature=reg_l2_feature)


# load compiled model, To compile model run compile_model.py
sm = pickle.load(open('<path_to_compiled_model>/FLDCRF_opportunity_HL.pkl', 'rb'))     
 

# train model and generate inferred quantities

fit = sm.optimizing(data=dat, algorithm='BFGS')

We also provide a forward inference code (online inference) inside the .stan file, described under the 'generated_quantities' block. We generate inferred probability values for each class and are saved under the 'Actual_proby' variable. We will include the Forward-backward inference shortly.

# save output as necessary, here we save the predicted class probability values [n_testvid, max_step, n_class]

HL = fit['Actual_proby']  

filename_new = ' '.join(('<path_to_save_file>', '.mat'))

scipy.io.savemat(filename_new.replace(" ", ""), {'HL': HL})

We are also in the process of applying FLDCRF in end-to-end models and its GPU implementation. Any contributions are welcome. We also plan to test FLDCRF on larger datasets.

Next updates:
1. Forward-backward inference
2. Multi-label FLDCRF code and data
3. GPU implementation
4. CNN-FLDCRF

If you find our codes useful, please cite our paper(s) [1, 6].

Email us for any queries on FLDCRF or our codes:
1. satyajit001@e.ntu.edu.sg
2. satyajitneogiju@gmail.com
3. jdauwels@ntu.edu.sg

References:
1. S. Neogi, M. Hoy, K. Dang, H. Yu, J. Dauwels, "Context Model for Pedestrian Intention Prediction using Factored Latent-Dynamic Conditional Random Fields", https://arxiv.org/pdf/1907.11881.pdf (2019).
2. L. P. Morency, A. Quattoni, T. Darrell, "Latent-Dynamic Discriminative Models for Continuous Gesture Recognition". In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, IEEE Computer Society, 2007.
3. R. Chavarriaga, H. Sagha, A. Calatroni, S. Digumarti, G. Trster, J. D. R. Milln, D Roggen.
"The Opportunity challenge: A benchmark database for on-body sensor-based activity
recognition". Pattern Recognition Letters, 2013.
4. R. C. B. Madeo, C. A. M. Lima, S. M. PERES, "Gesture Unit Segmentation using Support
Vector Machines: Segmenting Gestures from Rest Positions". In Proceedings of the 28th
Annual ACM Symposium on Applied Computing (SAC), p. 46-52, 2013.
5. I. Kotseruba, A. Rasouli, J. K. Tsotsos, “Joint Attention in Autonomous Driving (JAAD).” arXiv preprint arXiv:1609.04741(2016).
6. S. Neogi, M. Hoy, W. Chaoqun, J. Dauwels, \Context Based Pedestrian Intention Prediction
Using Factored Latent Dynamic Conditional Random Fields". In Proceed-
ings of the 2017 IEEE Symposium Series on Computational Intelligence (SSCI), DOI:
10.1109/SSCI.2017.8280970, 2017.
