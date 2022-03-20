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

        ### numerical variables (just assigning category type)
        num_columns = ['admission_source_code', 'time_in_hospital', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
                    'admission_type_code', 'discharge_disposition_code', 'num_lab_procedures', 'hemoglobin_level']
        for col in num_columns:
            _df[col] = _df[col].astype('float64')

        ### binary variables

        binary_cols = ['diuretics',
                       'insulin',
                       'change',
                       'diabetesMed',
                       'complete_vaccination_status']
        for col in binary_cols:
            _df[col] = _df[col].str.lower()

        _df = _df.assign(#has_prosthesis = _df['has_prosthesis'].map({True: 1, False: 0}),
                         #blood_transfusion = _df['blood_transfusion'].map({True: 1, False: 0}),
                         diuretics = _df['diuretics'].map({'yes': True, 'no': False}),            
                         insulin = _df['insulin'].map({'yes': True, 'no': False}),            
                         change = _df['change'].map({'ch': True, 'no': False}),
                         diabetesMed = _df['diabetesMed'].map({'yes': True, 'no': False}),
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
        _df['age'] = _df['age'].str.lower().astype('category')
        ordered_age = ['unknown', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        _df = _df.assign(age=_df['age'].cat.set_categories(ordered_age, ordered=True))
        _df.loc[~(_df['age'].isin(ordered_age)), 'age'] = 'unknown'

        #weight
        _df['weight'] = _df['weight'].astype('category')
        ordered_weight = ['unknown', '0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200', '>200']
        _df = _df.assign(weight=_df['weight'].cat.set_categories(ordered_weight, ordered=True))
        _df.loc[~(_df['weight'].isin(ordered_weight)), 'weight'] = 'unknown'

        #max_glucose
        _df['max_glu_serum'] = _df['max_glu_serum'].str.lower().astype('category')
        ordered_max_glu = ['unknown', 'norm', '>200', '>300']
        _df = _df.assign(max_glu_serum=_df['max_glu_serum'].cat.set_categories(ordered_max_glu, ordered=True))
        _df.loc[~(_df['max_glu_serum'].isin(ordered_max_glu)), 'max_glu_serum'] = 'unknown'

        #A1Cresult
        _df['A1Cresult'] = _df['A1Cresult'].str.lower().astype('category')
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
        transferred = [2, 3, 4, 5, 9, 10, 15, 22, 23, 24, 27, 28, 29, 12, 16, 17]
        died = [11, 19, 20, 21]
        home_services = [6, 8]
        all_cats = home+left_ama+hospice+transferred+died+home_services
        
        _df['discharge_disposition_code'] = _df['discharge_disposition_code'].fillna(value='unknown')
        _df.loc[~(_df['discharge_disposition_code'].isin(all_cats)), 'discharge_disposition_code'] = 'unknown'
        _df.loc[(_df['discharge_disposition_code'].isin(home)), 'discharge_disposition_code'] = 'discharged_home'
        _df.loc[(_df['discharge_disposition_code'].isin(left_ama)), 'discharge_disposition_code'] = 'left_ama'
        _df.loc[(_df['discharge_disposition_code'].isin(hospice)), 'discharge_disposition_code'] = 'discharged_hospice'
        _df.loc[(_df['discharge_disposition_code'].isin(transferred)), 'discharge_disposition_code'] = 'transferred_inpatient'
        _df.loc[(_df['discharge_disposition_code'].isin(died)), 'discharge_disposition_code'] = 'expired'
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
        
        # diagnosis columns risk
        all_codes = [str(code) for code in list(range(1, 900))] + ['e'+str(code) for code in list(range(800, 900))] +['v'+str(code) for code in list(range(1, 90))]
        
        diag_readmission = {'diag_1_risk': {'very_low': ['10', '11', '110', '115', '117', '131', '133', '135', '136', '141', '142', '143', '145', '147', '148', '149', '160', '161', '163', '164', '170', '172', '173', '175', '180', '184', '187', '192', '194', '207', '210', '212', '214', '215', '216', '217', '218', '219', '223', '226', '227', '228', '229', '236', '237', '240', '244', '245', '246', '261', '262', '266', '272', '273', '283', '3', '301', '304', '308', '314', '318', '320', '322', '323', '324', '325', '327', '334', '335', '336', '337', '342', '344', '345', '346', '35', '351', '353', '36', '360', '361', '362', '365', '366', '369', '370', '372', '373', '375', '376', '377', '378', '379', '380', '381', '383', '384', '385', '386', '388', '39', '395', '405', '41', '412', '416', '417', '430', '448', '454', '461', '462', '463', '47', '470', '474', '48', '483', '49', '495', '500', '501', '510', '513', '521', '522', '523', '524', '528', '529', '540', '542', '565', '566', '57', '579', '58', '582', '583', '601', '602', '603', '607', '610', '615', '616', '617', '618', '620', '621', '622', '623', '625', '632', '633', '634', '637', '640', '641', '645', '647', '649', '652', '653', '654', '655', '656', '657', '659', '66', '661', '663', '664', '665', '669', '671', '674', '683', '686', '692', '698', '7', '700', '703', '705', '708', '720', '725', '726', '732', '734', '735', '745', '746', '747', '75', '751', '759', '78', '793', '795', '797', '800', '802', '803', '804', '810', '814', '817', '827', '831', '832', '833', '834', '835', '836', '837', '838', '839', '84', '840', '843', '845', '847', '848', '850', '854', '862', '863', '864', '865', '866', '867', '868', '870', '871', '873', '875', '878', '879', '880', '881', '883', '885', '886', '891', '892', '893', '895', '897', 'v25', 'v26', 'v45', 'v51', 'v53', 'v54', 'v55', 'v56', 'v63', 'v66', 'v70', 'v71'], 'low': ['150', '151', '153', '154', '155', '156', '157', '158', '162', '174', '182', '185', '188', '189', '191', '193', '196', '197', '198', '201', '205', '211', '220', '225', '230', '233', '235', '238', '241', '250', '252', '253', '255', '274', '275', '276', '277', '278', '280', '284', '285', '288', '289', '290', '291', '292', '295', '296', '300', '303', '305', '307', '311', '331', '332', '333', '349', '354', '355', '357', '358', '368', '38', '394', '396', '398', '401', '402', '404', '410', '411', '413', '414', '415', '42', '421', '424', '425', '426', '427', '428', '431', '432', '433', '435', '436', '437', '441', '442', '446', '451', '453', '455', '456', '458', '459', '464', '465', '466', '473', '480', '481', '482', '485', '486', '487', '490', '491', '492', '493', '494', '496', '511', '515', '516', '518', '519', '527', '53', '530', '531', '532', '533', '534', '535', '550', '552', '553', '555', '556', '557', '558', '560', '562', '564', '569', '573', '574', '575', '576', '577', '578', '581', '584', '585', '590', '591', '592', '595', '596', '599', '600', '604', '608', '614', '626', '627', '642', '648', '658', '660', '680', '681', '682', '693', '695', '710', '711', '715', '716', '721', '722', '723', '724', '727', '728', '729', '730', '733', '736', '738', '780', '781', '782', '783', '784', '785', '786', '789', '79', '794', '799', '8', '801', '805', '807', '812', '813', '821', '822', '823', '824', '825', '844', '851', '852', '861', '88', '9', 'unknown', 'v57'], 'medium': ['112', '183', '199', '202', '203', '239', '242', '251', '282', '286', '287', '293', '294', '297', '298', '309', '312', '348', '403', '420', '423', '429', '434', '438', '440', '444', '445', '447', '452', '475', '478', '5', '507', '512', '514', '536', '537', '54', '541', '551', '568', '570', '571', '572', '593', '594', '611', '644', '694', '70', '707', '714', '717', '718', '719', '756', '787', '788', '790', '796', '808', '815', '816', '820', '853', '860', '882', '890'], 'medium_high': ['146', '152', '171', '179', '195', '200', '204', '208', '281', '306', '310', '338', '34', '340', '341', '350', '359', '374', '382', '397', '443', '457', '526', '567', '586', '588', '598', '619', '646', '685', '709', '737', '792', '806', '82', '826', '94', 'v58'], 'high': ['23', '263', '27', '271', '279', '31', '347', '352', '356', '506', '508', '52', '543', '580', '643', '696', '731', '753', 'v60']}, 'diag_2_risk': {'very_low': ['111', '115', '117', '123', '130', '131', '141', '145', '155', '164', '171', '173', '180', '182', '188', '192', '208', '215', '217', '218', '225', '226', '227', '228', '239', '240', '241', '246', '251', '252', '253', '256', '259', '266', '269', '27', '272', '273', '274', '275', '278', '289', '291', '299', '302', '306', '308', '31', '310', '314', '316', '317', '318', '322', '323', '324', '325', '327', '333', '336', '338', '34', '347', '35', '350', '351', '352', '353', '355', '356', '359', '360', '362', '365', '366', '369', '372', '373', '376', '378', '379', '380', '381', '383', '386', '388', '389', '395', '40', '412', '422', '423', '429', '430', '442', '448', '451', '455', '46', '460', '461', '462', '463', '464', '470', '472', '473', '475', '477', '478', '483', '485', '487', '490', '495', '501', '508', '510', '513', '517', '519', '52', '520', '521', '523', '524', '527', '534', '54', '540', '542', '543', '550', '568', '579', '588', '595', '598', '600', '602', '603', '604', '607', '610', '618', '621', '622', '623', '626', '627', '634', '641', '642', '644', '645', '647', '648', '649', '652', '656', '658', '659', '66', '661', '663', '664', '665', '674', '685', '686', '691', '694', '695', '698', '701', '702', '703', '704', '706', '713', '718', '725', '728', '729', '734', '737', '741', '742', '75', '750', '751', '755', '756', '758', '759', '78', '782', '783', '791', '793', '795', '797', '800', '801', '806', '807', '810', '814', '815', '816', '821', '822', '832', '833', '836', '837', '842', '843', '844', '847', '851', '852', '853', '861', '862', '863', '864', '865', '866', '868', '869', '870', '871', '873', '88', '880', '881', '882', '883', '884', '892', '893', 'v10', 'v11', 'v13', 'v14', 'v16', 'v18', 'v23', 'v46', 'v53', 'v55', 'v57', 'v66', 'v70', 'v72', 'v86'], 'low': ['135', '151', '162', '174', '185', '189', '196', '198', '199', '200', '204', '211', '233', '244', '250', '261', '262', '263', '276', '277', '279', '280', '282', '283', '284', '285', '286', '287', '288', '290', '294', '295', '296', '300', '301', '303', '304', '305', '309', '312', '331', '340', '344', '345', '346', '348', '349', '357', '358', '38', '382', '396', '397', '398', '401', '402', '403', '404', '41', '410', '411', '413', '414', '415', '416', '42', '420', '421', '424', '425', '426', '427', '428', '432', '434', '435', '436', '437', '438', '441', '443', '458', '459', '466', '481', '482', '486', '491', '492', '493', '496', '507', '511', '512', '514', '515', '518', '53', '530', '531', '535', '552', '553', '555', '556', '557', '558', '560', '562', '564', '565', '566', '567', '569', '571', '574', '575', '576', '578', '581', '583', '584', '585', '586', '590', '591', '592', '593', '599', '601', '611', '614', '616', '617', '620', '625', '680', '681', '682', '693', '696', '70', '707', '709', '710', '711', '714', '715', '716', '717', '719', '722', '723', '724', '726', '727', '730', '731', '733', '736', '746', '753', '780', '781', '784', '786', '787', '788', '789', '79', '792', '794', '799', '8', '802', '805', '812', '813', '823', '850', '860', '867', '94', 'unknown', 'v12', 'v15', 'v17', 'v42', 'v43', 'v45', 'v54', 'v58', 'v62', 'v65', 'v85'], 'medium': ['112', '138', '153', '154', '157', '172', '191', '197', '201', '203', '214', '238', '242', '255', '292', '293', '297', '298', '311', '319', '337', '342', '354', '368', '394', '431', '433', '440', '444', '446', '447', '452', '453', '456', '457', '465', '480', '494', '516', '528', '532', '533', '536', '537', '570', '573', '577', '594', '596', '684', '712', '721', '738', '745', '785', '790', '796', '820', '824', '840', '9', 'v49', 'v63', 'v64'], 'medium_high': ['11', '110', '150', '156', '183', '193', '202', '205', '220', '245', '260', '281', '307', '332', '335', '343', '454', '522', '572', '608', '619', '646', '705', '747', '808', '825', '831', '845', '891', 'v44'], 'high': ['114', '136', '152', '179', '186', '258', '320', '341', '377', '405', '474', '484', '500', '654', '692', '826', '894', '96', 'v61']}, 'diag_3_risk': {'very_low': ['11', '110', '122', '132', '136', '139', '14', '146', '151', '161', '163', '17', '170', '171', '172', '173', '175', '180', '182', '186', '191', '193', '195', '214', '216', '217', '220', '226', '227', '228', '233', '235', '239', '240', '243', '245', '246', '258', '259', '260', '265', '270', '271', '274', '289', '297', '299', '3', '306', '307', '308', '313', '317', '318', '323', '334', '335', '34', '347', '35', '350', '351', '353', '354', '355', '358', '359', '360', '361', '366', '369', '370', '372', '373', '374', '376', '377', '379', '381', '382', '384', '386', '387', '388', '395', '417', '421', '430', '431', '445', '448', '452', '454', '460', '463', '464', '465', '466', '47', '470', '472', '473', '475', '478', '480', '481', '483', '484', '485', '490', '495', '5', '500', '508', '510', '516', '523', '525', '527', '528', '529', '537', '54', '540', '542', '543', '556', '565', '566', '57', '579', '594', '603', '605', '610', '611', '614', '618', '620', '621', '622', '623', '624', '625', '626', '627', '641', '642', '643', '644', '646', '647', '649', '652', '653', '654', '655', '656', '657', '658', '659', '66', '661', '663', '664', '665', '670', '671', '684', '685', '690', '694', '697', '7', '701', '702', '703', '704', '712', '714', '717', '718', '720', '725', '726', '732', '734', '735', '736', '738', '741', '742', '746', '747', '75', '752', '754', '757', '758', '759', '78', '793', '795', '796', '800', '810', '811', '814', '815', '821', '822', '823', '825', '826', '831', '834', '836', '841', '842', '845', '847', '848', '850', '851', '853', '860', '862', '864', '865', '866', '868', '870', '871', '873', '875', '876', '877', '879', '88', '881', '882', '883', '884', '891', '893', '9', '94', 'v11', 'v13', 'v16', 'v17', 'v18', 'v22', 'v23', 'v25', 'v27', 'v53', 'v54', 'v55', 'v57', 'v61', 'v63', 'v70', 'v86'], 'low': ['112', '135', '153', '154', '157', '179', '185', '188', '189', '198', '201', '202', '203', '204', '205', '211', '218', '241', '242', '244', '250', '252', '253', '263', '266', '272', '275', '276', '277', '278', '280', '281', '282', '285', '286', '287', '288', '290', '291', '293', '294', '295', '296', '300', '301', '303', '305', '310', '311', '319', '327', '332', '333', '337', '338', '344', '345', '346', '348', '356', '357', '362', '368', '378', '38', '380', '389', '394', '397', '401', '402', '404', '41', '410', '411', '412', '413', '414', '415', '416', '423', '424', '425', '426', '427', '428', '429', '432', '433', '435', '437', '438', '440', '441', '442', '443', '446', '451', '455', '456', '457', '458', '459', '461', '462', '482', '486', '491', '492', '493', '494', '496', '501', '507', '512', '514', '515', '517', '518', '519', '53', '530', '531', '533', '535', '553', '555', '558', '560', '562', '564', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '578', '583', '584', '586', '588', '590', '591', '592', '593', '599', '600', '601', '616', '617', '648', '680', '682', '692', '693', '696', '698', '70', '705', '707', '709', '710', '713', '715', '716', '721', '722', '724', '728', '729', '730', '731', '733', '737', '745', '753', '780', '782', '783', '784', '786', '787', '788', '79', '790', '792', '794', '799', '802', '808', '812', '813', '824', '840', '861', '867', 'unknown', 'v10', 'v14', 'v15', 'v42', 'v43', 'v44', 'v45', 'v58', 'v62', 'v66', 'v72', 'v85'], 'medium': ['131', '138', '162', '174', '183', '196', '197', '199', '225', '238', '255', '284', '292', '298', '304', '309', '331', '336', '340', '342', '343', '349', '365', '383', '396', '398', '403', '405', '434', '436', '444', '447', '453', '477', '487', '511', '521', '532', '536', '577', '580', '582', '585', '595', '596', '598', '608', '619', '681', '708', '711', '719', '723', '727', '781', '785', '789', '791', '8', '805', '807', '820', '892', 'v12', 'v46', 'v49', 'v64', 'v65'], 'medium_high': ['141', '150', '155', '200', '208', '251', '256', '261', '262', '273', '279', '283', '312', '42', '420', '522', '550', '552', '557', '581', '604', '607', '660', '686', '695', '751', '756', '801', '816'], 'high': ['111', '117', '156', '158', '192', '215', '223', '236', '314', '341', '391', '506', '524', '534', '597', '602', '744', '755', '837', '838', '844', '852', '890', 'v60']}}

        _df['diag_1_risk'] = _df['diag_1']
        _df['diag_2_risk'] = _df['diag_2']
        _df['diag_3_risk'] = _df['diag_3']                         
        
        columns = ['diag_1_risk', 'diag_2_risk', 'diag_3_risk']
        for col in columns:
            _dict = diag_readmission[col]
            _df[col] = _df[col].str.lower().str[0:3]
            _df.loc[~(_df[col].isin(all_codes)), col] = 'unknown'
            
             
            _dict = diag_readmission[col]

            _df.loc[(_df[col].isin(_dict['very_low'])), col] = 'very_low'
            _df.loc[(_df[col].isin(_dict['low'])), col] = 'low'
            _df.loc[(_df[col].isin(_dict['medium'])), col] = 'medium'
            _df.loc[(_df[col].isin(_dict['medium_high'])), col] = 'medium_high'
            _df.loc[(_df[col].isin(_dict['high'])), col] = 'high'

            _df[col] = _df[col].astype('category')
            
        ordered_risk = ['unknown', 'very_low', 'low', 'medium', 'medium_high', 'high']
        _df = _df.assign(diag_1_risk=_df['diag_1_risk'].cat.set_categories(ordered_risk, ordered=True))
        _df = _df.assign(diag_2_risk=_df['diag_2_risk'].cat.set_categories(ordered_risk, ordered=True))
        _df = _df.assign(diag_3_risk=_df['diag_3_risk'].cat.set_categories(ordered_risk, ordered=True))
        
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
            _df[col] = _df[col].str.lower().str[0:3]
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
        _df['blood_type'] = _df['blood_type'].str.lower()
        _df['blood_type'].fillna(value='unknown')
        _df.loc[~(_df['blood_type'].isin(['a+', 'b+', 'o+', 'ab-', 'a-', 'o-', 'ab+', 'b-'])), 'blood_type'] = 'unknown'
        _df['blood_type'] = _df['blood_type'].astype('category')

        return _df