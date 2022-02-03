from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, power_transform, RobustScaler
import numpy as np

class custom_impute_scale(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, df, *args):
        return self
    
    def transform(self, df, *args):
       ### Put your transformation here
        _df = df.copy()
  
        num_features = ['time_in_hospital',
                        'num_lab_procedures',
                        'num_procedures',
                        'num_medications',
                        'number_outpatient',
                        'number_emergency',
                        'number_inpatient',
                        'number_diagnoses',
                        'hemoglobin_level']

        num_imputer = KNNImputer(n_neighbors = 5,
                         missing_values = np.nan,
                         weights = 'distance',
                         copy = True)

        robust_scaler = RobustScaler()

        _df[num_features] = num_imputer.fit_transform(_df[num_features])
        _df[num_features] = robust_scaler.fit_transform(_df[num_features])
                                          
        return _df