from sklearn.base import TransformerMixin
import category_encoders as ce

class custom_one_hot(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, df, *args):
        return self
    
    def transform(self, df, *args):
       ### Put your transformation here
        _df = df.copy()
        
        possible_cats = ['race',
                         'gender',
                         'admission_type_code',
                         'discharge_disposition_code',
                         'admission_source_code',
                         'payer_code',
                         'medical_specialty',
                         'diag_1',
                         'diag_2',
                         'diag_3',
                         'blood_type']
        
        cats = [cat for cat in _df.select_dtypes(include = 'category').columns.tolist() if cat in possible_cats]
        
        if len(cats) == 0:
            return _df
        
        oh = ce.OneHotEncoder(cols = cats,
                           handle_unknown='indicator',
                           handle_missing='indicator',
                           use_cat_names = True)

        return oh.fit_transform(_df)