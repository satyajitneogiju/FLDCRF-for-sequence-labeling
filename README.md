# FLDCRF-for-sequence-labeling
Factored Latent-Dynamic Conditional Random Fields for sequence labeling and prediction. FLDCRF outperforms LSTM on small/medium datasets (&lt;40,000 instances) across 6 experiments on 4 datasets.  

FLDCRF [1] is a single and multi-label generalization of LDCRF [2]. In our single-label experiments on 4 datasets (NTU [1], UCI opportunity [3], UCI gesture phase [4], JAAD [5]) across 6 experiments, FLDCRF improves over LDCRF performance and outperforms LSTM and LSTM-CRF on all test sets. 

Additionally, LSTM is known for tedious hyper-parameter optimization process, big-data-driven performance and long training time. FLDCRF offers easier model selection (with no need to tune number of epochs), performs better than LSTM on small/medium datasets, is much easier to comprehend and takes significantly lesser training time, even without GPU implementation. We also find FLDCRF to generalize better over validation and test sets, i.e., selected LSTM models perform worse on test sets than selected FLDCRF models, even though in some cases LSTM models perform better on validation. Such inconsistency across validation and test, blurry intuition and tedious model selection makes LSTM hard for industrial applications.


In our recent multi-label sequence labeling experiment on UCI opportunity dataset [3], FLDCRF outperformed LDCRF, Factorial CRF, LSTM-CRF, LSTM and a LSTM multi label model. We will update the paper link for multi-label experiment shortly.

To run our codes, you need to install pystan - 

pip install pystan

PyStan is the Python interface of Stan probabilistic modeling language. For pystan documentation, refer to this - https://pystan.readthedocs.io/en/latest/getting_started.html. For stan language documentation, please check this - https://mc-stan.org/users/documentation/. Stan provides easy-to-use BFGS, L-BFGS and Newton optimization algorithms with effective stopping criteria. As we do not need to tune the number of training epochs for FLDCRF, Pystan is a very useful platform.

Please download the Readme.rtf file for further instructions.  

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
1. S. Neogi, M. Hoy, K. Dang, H. Yu, J. Dauwels, "Context Model for Pedestrian Intention Prediction using Factored Latent-Dynamic Conditional Random Fields", https://arxiv.org/abs/1907.11881 (2019).
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
