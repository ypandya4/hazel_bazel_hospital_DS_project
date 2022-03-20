from sklearn.base import TransformerMixin
import category_encoders as ce

class custom_oe(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, df, *args):
        return self
    
    def transform(self, df, *args):
       ### Put your transformation here
        _df = df.copy()
  
        orde = ce.OrdinalEncoder(verbose=1,
                        cols=['age',
                               'weight',
                               'max_glu_serum',
                               'A1Cresult',
                               'complete_vaccination_status'],
                         handle_unknown='value',
                         handle_missing='value',
                         mapping = [{'col': 'age', 'mapping': {'0-10':1,
                                                               '10-20':2,
                                                               '20-30':3,
                                                               '30-40':4,
                                                               '40-50':5,
                                                               '50-60':6,
                                                               '60-70':7,
                                                               '70-80':8,
                                                               '80-90':9,
                                                               '90-100':10}},
        
                                    {'col': 'weight', 'mapping': {'0-25':1,
                                                                  '25-50':2, 
                                                                  '50-75':3,
                                                                  '75-100':4,
                                                                  '100-125':5, 
                                                                  '125-150':6, 
                                                                  '150-175':7, 
                                                                  '175-200':8, 
                                                                  '>200':9}},
                                    
                                    {'col': 'max_glu_serum', 'mapping': {'norm':1,
                                                                         '>200':2,
                                                                         '>300':3,}},
                                    
                                    {'col': 'A1Cresult', 'mapping': {'norm':1,
                                                                         '>7':2,
                                                                         '>8':3,}},
                                   
                                    {'col': 'complete_vaccination_status', 'mapping': {'Complete':1,
                                                                                       'Incomplete':0}},
                                   
                                    {'col': 'diag_1_risk', 'mapping': {'very_low':1,
                                                                  'low':2,
                                                                  'medium':3,
                                                                  'medium_high':4,
                                                                  'high':5}},
                                   
                                    {'col': 'diag_2_risk', 'mapping': {'very_low':1,
                                                                  'low':2,
                                                                  'medium':3,
                                                                  'medium_high':4,
                                                                  'high':5}},
                                   
                                    {'col': 'diag_3_risk', 'mapping': {'very_low':1,
                                                                  'low':2,
                                                                  'medium':3,
                                                                  'medium_high':4,
                                                                  'high':5}}])
                                          
        return orde.fit_transform(_df)