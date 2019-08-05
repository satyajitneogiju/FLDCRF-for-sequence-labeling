import pystan
import numpy as np
import scipy.io
import os
import pickle

sm = pystan.StanModel(file='<path>/FLDCRF_singlelabel.stan')

with open('<path>/FLDCRF_singlelabel.pkl', 'wb') as f:
    pickle.dump(sm, f)



