import pystan
import numpy as np
import scipy.io
import os
import pickle

sm = pystan.StanModel(file='<path>/FLDCRF_multilabel.stan')

with open('<path to save model>/FLDCRF_multilabel.pkl', 'wb') as f:
    pickle.dump(sm, f)