#%% import libs and read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X = df.drop('Survived', axis=1)
y = df.iloc[:, 1].values

# check info quickly
df.info()
df.head()
df.describe([.05,.95])
df.describe(include=['O'])
df.isnull().sum()



#%% Taking care of missing data
#drop PassId, Cabin, Name, Ticket and Age columns
X = X.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1)
X.info()

#impute missing values in Embarked and Age
df.groupby('Embarked').size()
X['Embarked'].fillna('S', inplace=True)
value_filled = X['Age'].mean()
X['Age'].fillna(value_filled, inplace=True)
X.isnull().sum()
#%% Engineering the SibSp and Parch
SibSp_mapping = {1: 0, 2: 0, 0: 1, 3: 1, 4:1, 5:1, 8:1}
X['SibSp2'] = X['SibSp'].map(SibSp_mapping)

Parch_mapping = {0: 0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}
X['Parch2'] = X['Parch'].map(Parch_mapping)

X = X.drop(['SibSp', 'Parch'], axis=1)

#%%convert categorical variables
from sklearn.preprocessing import LabelBinarizer
binarizer = LabelBinarizer()

binarizer.fit(X['Embarked'])    
Embarked_after = binarizer.transform(X['Embarked'])
Embarked_after = pd.DataFrame(Embarked_after, columns=['Port1', 'Port2', 'Port3'])

binarizer.fit(X['Sex'])    
Sex_after = binarizer.transform(X['Sex'])
Sex_after = pd.DataFrame(Sex_after, columns=['Sex2'])

X = pd.concat([X, Embarked_after, Sex_after], axis=1).drop(['Sex', 'Embarked'], axis=1)

#%%Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0    )

#%%
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#%% Logistic Regression 
#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score_log = accuracy_score(y_true=y_test,y_pred=y_pred)


#%% KNNs
# Fitting classifier to the Training set
# Create your classifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors= 5,metric='minkowski', p= 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score_KNN = accuracy_score(y_true=y_test,y_pred=y_pred)

#%% SVM
# Fitting classifier to the Training set
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state= 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score_SVM = accuracy_score(y_true=y_test,y_pred=y_pred)

#%% Kernel VSM
# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score_KernelSVM = accuracy_score(y_true=y_test,y_pred=y_pred)

#%% Best model
best_classifier = SVC(kernel='rbf', random_state=0)
best_classifier.fit(X_train, y_train)


#%% Fit to data_test and submit
df_test = pd.read_csv('test.csv')
X = df_test

# check info quickly
df_test.info()
df_test.head()
df_test.describe([.05,.95])
df_test.describe(include=['O'])
df_test.isnull().sum()


# Taking care of missing data
#drop PassId, Cabin, Name, Ticket and Age columns
X = X.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1)
X.info()

#impute missing values in Embarked and Age
df_test.groupby('Embarked').size()
X['Embarked'].fillna('S', inplace=True)
value_filled = X['Age'].mean()
X['Age'].fillna(value_filled, inplace=True)
value_filled = X['Fare'].median()
X['Fare'].fillna(value_filled, inplace=True)

X.isnull().sum()

#convert categorical variables
binarizer = LabelBinarizer()

binarizer.fit(X['Embarked'])    
Embarked_after = binarizer.transform(X['Embarked'])
Embarked_after = pd.DataFrame(Embarked_after, columns=['Port1', 'Port2', 'Port3'])

binarizer.fit(X['Sex'])    
Sex_after = binarizer.transform(X['Sex'])
Sex_after = pd.DataFrame(Sex_after, columns=['Sex2'])

X = pd.concat([X, Embarked_after, Sex_after], axis=1).drop(['Sex', 'Embarked'], axis=1)

#Feature Scaling
X = sc_X.transform(X)

#Predict the test set
y_pred = best_classifier.predict(X)
submission = df_test[['PassengerId']]
submission['Survived'] = y_pred
submission.to_csv('titanic_submission.csv', index=False)

