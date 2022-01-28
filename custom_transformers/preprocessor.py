from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class ColumnConverter(TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, df, *args):
        return self
    
    def transform(self, df, *args):
       ### Put your transformation here
        _df = df.copy()
       
        ### binary variables
        _df = _df.assign(has_prosthesis = _df['has_prosthesis'].map({True: 1, False: 0}),
                         blood_transfusion = _df['blood_transfusion'].map({True: 1, False: 0}),
                         diuretics = _df['diuretics'].map({'Yes': 1, 'No': 0}),
                         insulin = _df['insulin'].map({'Yes': 1, 'No': 0}),
                         change = _df['change'].map({'Ch': 1, 'No': 0}),
                         diabetesMed = _df['diabetesMed'].map({'Yes': 1, 'No': 0}))

        ### ordered categories
        _df['age'] = _df['age'].str.lstrip("\[").str.rstrip(")")
        _df['weight'] = _df['weight'].str.lstrip("\[").str.rstrip(")")
        _df['max_glu_serum'] = _df['max_glu_serum'].str.lower()
        _df['A1Cresult'] = _df['A1Cresult'].str.lower()
    
        #setting categories and replacing missing or unexpected values with 'unknown'
        #age
        _df['age'] = _df['age'].astype('category')
        ordered_age = ['unknown', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        _df = _df.assign(age=_df['age'].cat.set_categories(ordered_age, ordered=True))
        _df.loc[~(_df['age'].isin(ordered_age)), 'age'] = 'unknown'

        #weight
        _df['weight'] = _df['weight'].astype('category')
        ordered_weight = ['unknown', '0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200', '>200']
        _df = _df.assign(weight=_df['weight'].cat.set_categories(ordered_weight, ordered=True))
        _df.loc[~(_df['weight'].isin(ordered_weight)), 'weight'] = 'unknown'

        #max_glucose
        _df['max_glu_serum'] = _df['max_glu_serum'].astype('category')
        ordered_max_glu = ['unknown', 'norm', '>200', '>300']
        _df = _df.assign(max_glu_serum=_df['max_glu_serum'].cat.set_categories(ordered_max_glu, ordered=True))
        _df.loc[~(_df['max_glu_serum'].isin(ordered_max_glu)), 'max_glu_serum'] = 'unknown'

        #A1Cresult
        _df['A1Cresult'] = _df['A1Cresult'].astype('category')
        ordered_A1C = ['unknown', 'norm', '>7', '>8']
        _df = _df.assign(A1Cresult=_df['A1Cresult'].cat.set_categories(ordered_A1C, ordered=True))
        _df.loc[~(_df['A1Cresult'].isin(ordered_A1C)), 'A1Cresult'] = 'unknown'

        #Gender
        _df['gender'] = _df['gender'].astype('category')
        _df['gender'] = _df['gender'].str.lower()
        valid_genders = ['male', 'female']
        _df.loc[~(_df['gender'].isin(valid_genders)), 'gender'] = 'unknown'

        #race
        _df['race'] = _df['race'].astype('category')
        _df['race'] = _df['race'].str.lower().str.lstrip().str[0:3]
        black = ['afr', 'bla']
        white = ['cau', 'whi', 'eur']
        hispanic = ['his', 'lat']
        asian = ['asi']
        race_options = black+white+hispanic+asian
        _df.loc[~(_df['race'].isin(race_options)), 'race'] = 'unknown/other'
        _df.loc[(_df['race'].isin(black)), 'race'] = 'black'
        _df.loc[(_df['race'].isin(white)), 'race'] = 'white'
        _df.loc[(_df['race'].isin(hispanic)), 'race'] = 'hispanic'
        _df.loc[(_df['race'].isin(asian)), 'race'] = 'asian'

        #insurance status (determined from payer code)
        _df.payer_code = _df.payer_code.where(_df.payer_code != '?')
        _df.payer_code = _df.payer_code.fillna(value='unknown')
        _df.loc[(_df['payer_code'] == 'SP'), 'payer_code'] == 'uninsured'
        _df.loc[~(_df['payer_code'].isin(['SP', 'unknown'])), 'payer_code'] = 'insured'
        _df['payer_code'] = _df['payer_code'].astype('category')

        #admission type
        _df.loc[(_df['admission_type_code'].isin([5, 6, 8])), 'admission_type_code'] = 'n/a'
        _df.loc[(_df['admission_type_code'] == 1), 'admission_type_code'] = 'emergency'
        _df.loc[(_df['admission_type_code'] == 2), 'admission_type_code'] = 'urgent'
        _df.loc[(_df['admission_type_code'] == 3), 'admission_type_code'] = 'elective'
        _df.loc[(_df['admission_type_code'] == 4), 'admission_type_code'] = 'newborn'
        _df.loc[(_df['admission_type_code'] == 7), 'admission_type_code'] = 'trauma'
        _df['admission_type_code'] = _df['admission_type_code'].astype('category')

        #admission source

        #discharge code
        home = [1]
        left_ama = [7]
        hospice = [13, 14]
        transferred = [2, 3, 4, 5, 9, 10, 15, 22, 23, 24, 27, 28, 29]
        died = [11, 19, 20, 21]
        for_outpatient_services = [12, 16, 17]
        home_services = [6, 8]
        all_cats = home+left_ama+hospice+transferred+died+for_outpatient_services+home_services

        _df.loc[~(_df['discharge_disposition_code'].isin(all_cats)), 'discharge_disposition_code'] = 'unknown'
        _df.loc[(_df['discharge_disposition_code'].isin(home)), 'discharge_disposition_code'] = 'discharged_home'
        _df.loc[(_df['discharge_disposition_code'].isin(left_ama)), 'discharge_disposition_code'] = 'left_ama'
        _df.loc[(_df['discharge_disposition_code'].isin(hospice)), 'discharge_disposition_code'] = 'discharged_hospice'
        _df.loc[(_df['discharge_disposition_code'].isin(transferred)), 'discharge_disposition_code'] = 'transferred_inpatient'
        _df.loc[(_df['discharge_disposition_code'].isin(died)), 'discharge_disposition_code'] = 'expired'
        _df.loc[(_df['discharge_disposition_code'].isin(for_outpatient_services)), 'discharge_disposition_code'] = 'transferred_outpatient'
        _df.loc[(_df['discharge_disposition_code'].isin(home_services)), 'discharge_disposition_code'] = 'home_care'
        _df['discharge_disposition_code'] = _df['discharge_disposition_code'].astype('category')
        
        #medical specialty (only over 100 patients)
        selected_specialties = ['pulmonology',
                                'internalmedicine', 
                                'cardiology', 
                                'unknown', 
                                'surgery-general', 
                                'emergency/trauma',
                                'physicalmedicineandrehabilitation', 
                                'family/generalpractice',
                                'surgery-cardiovascular/thoracic',
                                'nephrology',
                                'radiologist',
                                'hematology/oncology', 
                                'other', 
                                'orthopedics',
                                'orthopedics-reconstructive', 
                                'pediatrics-endocrinology',
                                'gastroenterology', 
                                'surgery-vascular', 
                                'obstetricsandgynecology',
                                'psychiatry', 
                                'urology', 
                                'surgery-neuro', 
                                'oncology', 
                                'neurology',
                                'pediatrics']
        
        _df['medical_specialty'] = _df['medical_specialty'].str.lower()
        _df.loc[(_df['medical_specialty'] == '?'), 'medical_specialty'] = 'unknown'
        _df.loc[~(_df['medical_specialty'].isin(selected_specialties)), 'medical_specialty'] = 'other'
        _df['medical_specialty'] = _df['medical_specialty'].astype('category')
       

        return _df