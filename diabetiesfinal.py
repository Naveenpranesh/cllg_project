import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import metrics
df_pandas = pd.read_csv('../Dataset/diabetes.csv')
df_pandas.head()
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
df_pandas.info()
missing_values_table(df_pandas)
diabetes_map = {1:1, 0:2}
df_pandas['Outcome'] = df_pandas['Outcome'].map(diabetes_map)
# mark zero values as missing or NaN
df_pandas = df_pandas.replace(0, np.NaN)
# fill missing values with mean column values
df_pandas.fillna(df_pandas.mean(), inplace=True)
diabetes_map = {1:1, 2:0}
df_pandas['Outcome'] = df_pandas['Outcome'].map(diabetes_map)
# count the number of NaN values in each column
print(df_pandas.isnull().sum())
df_nn = df_pandas
df_pandas.head()
sns.heatmap(df_pandas[df_pandas.columns[:8]].corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
sns.countplot(x='Outcome',data=df_pandas)
plt.show()
columns=df_pandas.columns[:8]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    df_pandas[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
diab1=df_pandas[df_pandas['Outcome']==1]
columns=df_pandas.columns[:8]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    diab1[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
sns.pairplot(data=df_pandas,hue='Outcome',diag_kind='hist')
plt.show()




###   Gradient Boosting
dataset = pd.read_csv('../Dataset/diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import GradientBoostingClassifier
gbe = GradientBoostingClassifier(random_state=42)
parameters={'learning_rate': [0.05, 0.1, 0.5],
            'max_features': [0.5, 1],
            'max_depth': [3, 4, 5]
}
gridsearch=GridSearchCV(gbe, parameters, cv=100, scoring='roc_auc')
gridsearch.fit(X, y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
gbi = GradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                 max_features=0.5,
                                 random_state=42)
X_train,X_test,y_train, y_test = train_test_split(X, y, random_state=42)
gbi.fit(X_train, y_train)
y_pred = gbi.predict_proba(X_test)[:,1]
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))


###   Apply Feed Forward Neural Network & Evaulate Results.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
dl_epoch = 100													
model = Sequential()
model.add(Dense(500, input_dim=8,init='uniform' ,activation='relu'))
model.add(Dense(100, init='uniform',activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,y_train, epochs=dl_epoch, batch_size=70, validation_data=(X_test, y_test))
plt.rcParams["figure.figsize"] = [20, 10]

for key in history.history.keys():
    plt.plot(range(1, dl_epoch+1), history.history[key])

plt.legend(list(history.history.keys()), loc='upper left')
plt.title('Key Matrics vs Epochs for Diabetes Data')
plt.show()

###   ADABOOST CLASSIFIER

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))
###   Apply Feed Forward Neural Network & Evaulate Results.

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
dl_epoch = 100

model = Sequential()
model.add(Dense(500, input_dim=8,init='uniform' ,activation='relu'))
model.add(Dense(100, init='uniform',activation='relu'))
model.add(Dense(1, init='uniform',activation='sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,y_train, epochs=dl_epoch, batch_size=70, validation_data=(X_test, y_test))

plt.rcParams["figure.figsize"] = [20, 10]

for key in history.history.keys():
    plt.plot(range(1, dl_epoch+1), history.history[key])

plt.legend(list(history.history.keys()), loc='upper left')
plt.title('Key Matrics vs Epochs for Diabetes Data')
plt.show()
