import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.ensemble import RandomForestClassifier

def read_data():
    df = pd.read_csv("dataset.csv")
    return df


def data_preprocessing(df):
    # Feature engineering
    df1 = df[['SeniorCitizen', 'Partner', 'Dependents',\
    'tenure','PhoneService', 'MultipleLines', 'InternetService',\
    'OnlineSecurity','Churn']]

    # Missing values treatment
    df1.dropna(inplace=True)

    # Label encoding
    encs = {}
    for col in df1.columns:
        if df1[col].dtype == "object":
            encs[col] = LabelEncoder()
            df1[col]   =encs[col].fit_transform(df1[col])
            

    # train test split 
    x = df1.iloc[:,:-1]
    y = df1['Churn']    
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=42) 

    # Standardization
    sc_train = StandardScaler()
    sc_test = StandardScaler()
    X_train = sc_train.fit_transform(X_train)
    X_test = sc_test.fit_transform(X_test)

    return X_train, X_test, y_train, y_test
            
def logestic_training(X_train, y_train):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    return clf

def knn_training(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    return clf    

def tree_training(X_train, y_train):
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    return clf    

def forest_training(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(X_train, y_train)
    return clf    

def test_classifier(clf,X_test , y_test):
    score = clf.score( X_test , y_test) 
    return score   

def algo_eval(clf ,X_test , y_test):
    y_pred = clf.predict(X_test)
    new_df = classification_report(y_test, y_pred)
    return new_df 



def row_probability(clf,attrs):
    r_pred = clf.predict_proba(attrs)
    return r_pred
    





# df = read_data()
# X_train, X_test, y_train, y_test = data_preprocessing(df)
# c = knn_training(X_train,y_train) 
# res = test_classifier (c,X_test , y_test)
# print(res)
# x1 = algo_eval(c ,X_test , y_test)
# lines = x1.split('\n')
# for x in lines:
#     st = x.split(' ')
#     print(st)
# mmm =10
# print(len(lines)) 

# c2 = tree_training(X_train,y_train)
# res = test_classifier (c2,X_test , y_test)
# print(res)

# c3 = logestic_training(X_train,y_train)
# res = test_classifier (c3,X_test , y_test)
# print(res)
