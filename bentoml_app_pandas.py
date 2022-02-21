import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame

import pickle
import numpy as np
import pandas as pd

# Load model
classifier = bentoml.sklearn.load_runner("loan_status_svm:latest")

# Create service with the model
service = bentoml.Service("loan_status_svm ", runners=[classifier])