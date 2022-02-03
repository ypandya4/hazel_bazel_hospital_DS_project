from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, power_transform, RobustScaler
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
        _df = _df.assign(#has_prosthesis = _df['has_prosthesis'].map({True: 1, False: 0}),
                         #blood_transfusion = _df['blood_transfusion'].map({True: 1, False: 0}),
                         diuretics = _df['diuretics'].map({'Yes': True, 'No': False}),            
                         insulin = _df['insulin'].map({'Yes': True, 'No': False}),            
                         change = _df['change'].map({'Ch': True, 'No': False}),
                         diabetesMed = _df['diabetesMed'].map({'Yes': True, 'No': False}),
                         complete_vaccination_status = _df['complete_vaccination_status'].map({'Complete': True, 'Incomplete': False}))
            
        _df['diuretics'] = _df['diuretics'].astype('bool')
        _df['insulin'] = _df['insulin'].astype('bool')
        _df['change'] = _df['change'].astype('bool')
        _df['diabetesMed'] = _df['diabetesMed'].astype('bool')
        _df['complete_vaccination_status'] = _df['complete_vaccination_status'].astype('bool')

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
        _df['gender'] = _df['gender'].str.lower()
        valid_genders = ['male', 'female']
        _df.loc[~(_df['gender'].isin(valid_genders)), 'gender'] = 'unknown'
        _df['gender'] = _df['gender'].astype('category')

        #race
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
        _df['race'] = _df['race'].astype('category')

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
        referral = [1, 2, 3]
        transfer = [4, 5, 6, 10, 18, 19, 22, 25]
        emergency = [7]
        all_admissions = referral+transfer+emergency

        _df.loc[~(_df['admission_source_code'].isin(all_admissions)), 'admission_source_code'] = 'unknown'
        _df.loc[(_df['admission_source_code'].isin(referral)), 'admission_source_code'] = 'referral'
        _df.loc[(_df['admission_source_code'].isin(transfer)), 'admission_source_code'] = 'transfer'
        _df.loc[(_df['admission_source_code'].isin(emergency)), 'admission_source_code'] = 'emergency'
        _df['admission_source_code'] = _df['admission_source_code'].astype('category')

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
       
       # diagnosis columns
        infection = [str(code) for code in list(range(1, 140))]
        neoplasms = [str(code) for code in list(range(141, 240))]
        endocrine_nutritional_metabolic_immune = [str(code) for code in list(range(241, 280))]
        blood = [str(code) for code in list(range(280, 290))]
        mental = [str(code) for code in list(range(290, 320))]
        nervous_system = [str(code) for code in list(range(320, 390))]
        circulatory = [str(code) for code in list(range(390, 460))]
        respiratory = [str(code) for code in list(range(460, 520))]
        digestive = [str(code) for code in list(range(520, 580))]
        genitourinary = [str(code) for code in list(range(580, 630))]
        pregnancy = [str(code) for code in list(range(630, 680))]
        skin = [str(code) for code in list(range(680, 710))]
        musculoskeletal = [str(code) for code in list(range(710, 740))]
        congenital = [str(code) for code in list(range(740, 760))]
        perinatal = [str(code) for code in list(range(760, 780))]
        ill_defined = [str(code) for code in list(range(780, 800))]
        injury_poisoning = [str(code) for code in list(range(800, 900))] + ['e'+str(code) for code in list(range(800, 900))]
        supplemental = ['v'+str(code) for code in list(range(1, 90))]

        all_codes = infection + neoplasms + endocrine_nutritional_metabolic_immune + blood + mental+ nervous_system + circulatory + respiratory + digestive + genitourinary + pregnancy + skin + musculoskeletal+ congenital+ perinatal+ ill_defined+ injury_poisoning+ supplemental 

        columns = ['diag_1', 'diag_2', 'diag_3']
        for col in columns:
            _df[col] = _df[col].fillna(value='unknown')
            _df.loc[~(_df[col].isin(all_codes)), col] = 'unknown'
    
            _df.loc[(_df[col].isin(infection)), col] = 'infection'
            _df.loc[(_df[col].isin(neoplasms)), col] = 'neoplasms'
            _df.loc[(_df[col].isin(endocrine_nutritional_metabolic_immune)), col] = 'endocrine_nutritional_metabolic_immune'
            _df.loc[(_df[col].isin(blood)), col] = 'blood'
            _df.loc[(_df[col].isin(mental)), col] = 'mental'
            _df.loc[(_df[col].isin(nervous_system)), col] = 'nervous_system'
            _df.loc[(_df[col].isin(circulatory)), col] = 'circulatory'
            _df.loc[(_df[col].isin(respiratory)), col] = 'respiratory'
            _df.loc[(_df[col].isin(digestive)), col] = 'digestive'
            _df.loc[(_df[col].isin(genitourinary)), col] = 'genitourinary'
            _df.loc[(_df[col].isin(pregnancy)), col] = 'pregnancy'
            _df.loc[(_df[col].isin(skin)), col] = 'skin'
            _df.loc[(_df[col].isin(musculoskeletal)), col] = 'musculoskeletal'
            _df.loc[(_df[col].isin(congenital)), col] = 'congenital'
            _df.loc[(_df[col].isin(perinatal)), col] = 'perinatal'
            _df.loc[(_df[col].isin(ill_defined)), col] = 'ill_defined'
            _df.loc[(_df[col].isin(injury_poisoning)), col] = 'injury_poisoning'
            _df.loc[(_df[col].isin(supplemental)), col] = 'supplemental'
            _df.loc[(_df[col].isin(infection)), col] = 'infection'
            
            _df[col] = _df[col].astype('category')
        
        # blood type
        _df['blood_type'] = _df['blood_type'].astype('category')

        return _df