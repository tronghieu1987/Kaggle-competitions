#%%
#Import libs and read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('training_data.csv')
df_test = pd.read_csv('(name)_score.csv')
df_add = df_test[df_test['survival_1_year'] == 0]
df_add['survival_7_years'].fillna(0, inplace=True)
df_train = pd.concat([df_train, df_add], axis=0, ignore_index=True)

#%% Function to plot chart and Survival rate against each category
def chart_against_category(df, category):
    data_grouped = df.groupby([category, 'survival_7_years']).size()
    data_grouped= data_grouped.unstack()
    data_grouped.columns = ['Died', 'Survived']
    data_grouped['Perc. Died'] = data_grouped['Died'] / data_grouped[['Died', 'Survived']].sum(axis=1)
    data_grouped.plot.bar(color=['red', 'blue'])
    return data_grouped

#%% Handle 'symptoms'
from functools import partial

#Function to map 'symptoms' column to 0/1
def check_symptom(symptom, letter):
    return 1 if letter in str(symptom) else 0

#Get the set of all symptom letter
def symptoms_to_columns(df):
    list_symptoms = list(df['symptoms'])
    all_letter_symptom = set()
    for string in list_symptoms:
        for c in str(string):
            if c.isupper(): all_letter_symptom.add(c)
#    Just group based on 'O' and 'P' and the rest
    all_letter_symptom = ['O', 'P',]
    #Add additional columns according to letter_symptom
    for letter in all_letter_symptom:
        add_symptom_column = partial(check_symptom, letter=letter)
        df['Is' + letter] = df['symptoms'].apply(add_symptom_column)
    return df

df_train = symptoms_to_columns(df_train)
df_test = symptoms_to_columns(df_test)

# Count # of symptoms
def count_symptoms(value):
    return (1 + str(value).count(','))
df_train['symptoms'] = df_train['symptoms'].map(count_symptoms)
df_test['symptoms'] = df_test['symptoms'].map(count_symptoms)


#%% Features engeering the variables
# t_score: 2 groups (T1, T2) and (T3, T4)
def convert_t_score(value):
    return 0 if int(value[1]) in (1,2) else 1
df_train['t_score'] = df_train['t_score'].map(convert_t_score)
df_test['t_score'] = df_test['t_score'].map(convert_t_score)

# n_score: 2 groups (NX and N0) and N1
n_score_mapping = {'NX': 0, 'N0': 0, 'N1': 1}
df_train['n_score'] = df_train['n_score'].map(n_score_mapping)
df_test['n_score'] = df_test['n_score'].map(n_score_mapping)

# m_score: 2 groups M0 and the rest (M1a,b,c)
m_score_mapping = {'M0': 0, 'M1a': 1, 'M1b': 1, 'M1c': 1}
df_train['m_score'] = df_train['m_score'].map(m_score_mapping)
df_test['m_score'] = df_test['m_score'].map(m_score_mapping)

# Conver Weight and Height into BMI index
df_train['BMI'] = 703 * df_train['weight'] / (df_train['height'])**2
df_test['BMI'] = 703 * df_test['weight'] / (df_test['height'])**2

def map_BMI(bmi):
    return 1 if bmi > 30 else 0
df_train['BMIOver'] = df_train['BMI'].map(map_BMI)
df_test['BMIOver'] = df_test['BMI'].map(map_BMI)

df_train = df_train.drop(['weight', 'height', 'BMI'], axis=1)
df_test = df_test.drop(['weight', 'height', 'BMI'], axis=1)

#Conver 'race' into category (Race 2 and 3 same group)
race_mapping = {1: 'I', 2: 'II', 3: 'II', 4: 'IV'}
df_train['race'] = df_train['race'].map(race_mapping)
df_test['race'] = df_test['race'].map(race_mapping)

#Conver 'stage' into numeric to better reflect linear relationship
stage_mapping = {'I': 1, 'IIA': 2, 'IIB': 4, 'III': 3, 'IV':5}
df_train['stage'] = df_train['stage'].map(stage_mapping)
df_test['stage'] = df_test['stage'].map(stage_mapping)

#%% Treat missing data
#Treat '0' as NaN in some cols
df_train['psa_1_year'] = df_train['psa_1_year'].replace(0, np.nan)
df_train['tumor_1_year'] = df_train['tumor_1_year'].replace(0, np.nan)
df_test['psa_1_year'] = df_test['psa_1_year'].replace(0, np.nan)
df_test['tumor_1_year'] = df_test['tumor_1_year'].replace(0, np.nan)

#Function to fill predicted missing values into original df
def handling_missing_data(df, missing_var, list_split, categorical_var):
    df_split = df[list_split]
    # Convert categorical variables
    from sklearn.preprocessing import LabelBinarizer
    binarizer = LabelBinarizer()
    for col in categorical_var:
        binarizer.fit(df_split[col])
        col_after = pd.DataFrame(binarizer.transform(df_split[col]))
        df_split = pd.concat([df_split, col_after], axis=1).drop(col, axis=1)
    # Split into train and test
    train_split = df_split[df_split[missing_var].notnull()]
    test_split = df_split[df_split[missing_var].isnull()]
    
    X_train = train_split.iloc[:, 2:].values
    y_train = train_split.iloc[:, 0].values
    X_test = test_split.iloc[:, 2:].values
    train_idANDmissing = train_split.iloc[:, :2]
    test_id = test_split.iloc[:, 1]
    
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    # Fit Random Forest to predict the missing values
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 300)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    #add missing_pred to original df
    missing_fitted = pd.DataFrame(test_id)
    missing_fitted[missing_var] = y_pred
    missing_fitted = pd.concat([missing_fitted, train_idANDmissing], axis=0)
    missing_fitted = missing_fitted.sort_index()
    df[missing_var] = missing_fitted[missing_var]
    return df

# Handle missing in gleason_score and tumor_diagnosis
missing_vars = ['gleason_score', 'tumor_diagnosis']
for missing_var in missing_vars:
    list_split = [missing_var, 'id', 't_score', 'n_score', 'm_score', 'stage']
    categorical_var = ['stage']
    df_train = handling_missing_data(df_train, missing_var, list_split, categorical_var)
    df_test = handling_missing_data(df_test, missing_var, list_split, categorical_var)
    
# Handle missing in psa_diagnosis
missing_var = 'psa_diagnosis'
list_split = [missing_var, 'id', 't_score', 'n_score', 'm_score', 'stage','gleason_score', 'tumor_diagnosis']
categorical_var = ['stage']
df_train = handling_missing_data(df_train, missing_var, list_split, categorical_var)
df_test = handling_missing_data(df_test, missing_var, list_split, categorical_var)

# Handle missing in tumor_1_year and psa_1_year
missing_vars = ['tumor_1_year', 'psa_1_year']
for missing_var in missing_vars:
    list_split = [missing_var, 'id', 't_score', 'n_score', 'm_score', 'stage','gleason_score', 'tumor_diagnosis', 'psa_diagnosis']
    categorical_var = ['stage']
    df_train = handling_missing_data(df_train, missing_var, list_split, categorical_var)
    df_test = handling_missing_data(df_test, missing_var, list_split, categorical_var)

#%%
#drop cols with too many missing values and Date (irelevant)
to_drop = ['tumor_6_months', 'psa_6_months', 'diagnosis_date']
df_train.drop(to_drop, axis=1, inplace=True)
df_test.drop(to_drop, axis=1, inplace=True)

# Convert numeric cols to supposed categorical cols
to_object = ['race', 'previous_cancer', 'smoker', 'rd_thrpy', 'h_thrpy', 'chm_thrpy',\
             'cry_thrpy', 'brch_thrpy', 'rad_rem', 'multi_thrpy', 'survival_7_years']
df_train[to_object] = df_train[to_object].astype(object)
df_test[to_object] = df_test[to_object].astype(object)


#%% Handling missing data
#imputing the other missing cols
from sklearn.preprocessing import Imputer
imp_numeric = Imputer(strategy='mean', axis=0)

def impute(df):
    survival = df['survival_7_years']
    df.drop('survival_7_years', axis=1, inplace=True)
    
    df_numeric = df.select_dtypes(exclude=['O'])
    df_categorical = df.select_dtypes(include=['O'])
    
    imp_numeric.fit(df_numeric)
    df_numeric= pd.DataFrame(imp_numeric.transform(df_numeric), columns=df_numeric.columns)
    
#    fillna in categorical with most_frequent value
    df_categorical = df_categorical.apply(lambda x:x.fillna(x.value_counts().index[0]))
    df_categorical = df_categorical.astype(object)
    
    return pd.concat([df_numeric, df_categorical, survival], axis=1)

df_train = impute(df_train)
df_test = impute(df_test)

#%% Pick variables to be used in the model
var_not_modelled = ['id','survival_1_year', 'survival_7_years','family_history', 'smoker', 'tea', 'side']
X = df_train.drop(var_not_modelled, axis=1)
y = df_train[['survival_1_year','survival_7_years']]
#%% Encode categorical variables
from sklearn.preprocessing import LabelBinarizer
binarizer = LabelBinarizer()

#get categorical cols that are not in 0/1 yet
categorical_var = []
for col in X.columns:
    try:
        X.iloc[0][col] >= 0
    except TypeError:
        categorical_var.append(col)

#Convert into nemric cols
for col in categorical_var:
    binarizer.fit(X[col])
    col_after = pd.DataFrame(binarizer.transform(X[col]))
    X = pd.concat([X, col_after], axis=1).drop(col, axis=1)

#%% TRAIN AND FIT MODELS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

simulation_times = 15
all_confusion_matrix = dict()
all_result = dict()
result = []

#Splitting the dataset into the Training set and Test set and Feature scaling
for i in range(simulation_times):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    y_train = y_train.drop('survival_1_year', axis=1).astype(int).values.ravel()
#    split those who already died after 1 year
    y_test_died1year = y_test['survival_1_year'].astype(int).values.ravel()
    y_test = y_test.drop('survival_1_year', axis=1).astype(int).values.ravel()
    
    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    # Logistic Regression 
    #Fitting Logistic Regression to the Training set
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
#    Match y_pred and y_already died
    j = 0
    for item in y_test_died1year:
        if item == 0: y_pred[j] = 0
        j+=1
        
#    check_mismatch = pd.DataFrame({'Pred': y_pred, '1_year': y_test_died1year}, columns=['Pred', '1_year'])
#    mismatch = check_mismatch[(check_mismatch['1_year'] == 0) & (check_mismatch['Pred'] != 0)]
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    #Scoring
    score_log = accuracy_score(y_true=y_test,y_pred=y_pred)
    result.append(score_log)
    
all_result['Logistic'] = [np.mean(result), np.std(result)]
all_confusion_matrix['Logistic'] = cm

#%% KNN method
result = []
for i in range(simulation_times):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    y_train = y_train.drop('survival_1_year', axis=1).astype(int).values.ravel()
    y_test_died1year = y_test['survival_1_year'].astype(int).values.ravel()
    y_test = y_test.drop('survival_1_year', axis=1).astype(int).values.ravel()
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    j = 0
    for item in y_test_died1year:
        if item == 0: y_pred[j] = 0
        j+=1
    cm = confusion_matrix(y_test, y_pred)
    score_log = accuracy_score(y_true=y_test,y_pred=y_pred)
    result.append(score_log)
all_result['KNN'] = [np.mean(result), np.std(result)]
all_confusion_matrix['KNN'] = cm

#%% SVC method
result = []
for i in range(simulation_times):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    y_train = y_train.drop('survival_1_year', axis=1).astype(int).values.ravel()
    y_test_died1year = y_test['survival_1_year'].astype(int).values.ravel()
    y_test = y_test.drop('survival_1_year', axis=1).astype(int).values.ravel()
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    j = 0
    for item in y_test_died1year:
        if item == 0: y_pred[j] = 0
        j+=1
    cm = confusion_matrix(y_test, y_pred)
    score_log = accuracy_score(y_true=y_test,y_pred=y_pred)
    result.append(score_log)
all_result['SVC'] = [np.mean(result), np.std(result)]
all_confusion_matrix['SVC'] = cm

#%% Decision TRee method
result = []
for i in range(simulation_times):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    y_train = y_train.drop('survival_1_year', axis=1).astype(int).values.ravel()
    y_test_died1year = y_test['survival_1_year'].astype(int).values.ravel()
    y_test = y_test.drop('survival_1_year', axis=1).astype(int).values.ravel()
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    j = 0
    for item in y_test_died1year:
        if item == 0: y_pred[j] = 0
        j+=1
    cm = confusion_matrix(y_test, y_pred)
    score_log = accuracy_score(y_true=y_test,y_pred=y_pred)
    result.append(score_log)
all_result['DecisionTree'] = [np.mean(result), np.std(result)]
all_confusion_matrix['DecisionTree'] = cm

#%% Random Forest method
result = []
for i in range(simulation_times):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)
    y_train = y_train.drop('survival_1_year', axis=1).astype(int).values.ravel()
    y_test_died1year = y_test['survival_1_year'].astype(int).values.ravel()
    y_test = y_test.drop('survival_1_year', axis=1).astype(int).values.ravel()
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=300, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    j = 0
    for item in y_test_died1year:
        if item == 0: y_pred[j] = 0
        j+=1
    cm = confusion_matrix(y_test, y_pred)
    score_log = accuracy_score(y_true=y_test,y_pred=y_pred)
    result.append(score_log)
all_result['RandomForest'] = [np.mean(result), np.std(result)]
all_confusion_matrix['RandomForest'] = cm

#%% BEST MODEL: SVC

X = df_test.drop(var_not_modelled, axis=1)
# Encode categorical variables
binarizer = LabelBinarizer()

#Convert into nemric cols
for col in categorical_var:
    binarizer.fit(X[col])
    col_after = pd.DataFrame(binarizer.transform(X[col]))
    X = pd.concat([X, col_after], axis=1).drop(col, axis=1)

X = sc_X.transform(X)
best_classifier = SVC()
best_classifier.fit(X_train, y_train)

y_pred = best_classifier.predict(X)

#Match Survival_1_year in test_data to y_pred
submission = pd.read_csv('(name)_score.csv')

def map_surv_1_year(value):
    return value if value in(0,1) else np.nan
y_test_died1year = submission['survival_1_year'].map(map_surv_1_year).tolist()

j = 0
for item in y_test_died1year:
    if item == 0: y_pred[j] = 0
    j+=1

submission['survival_7_years'] = y_pred
submission.to_csv('(HieuTran)_score.csv', index=False)
