import numpy as np
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
###############################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump

def main():

    raw_data_Y = pd.read_csv('dane2.csv')
    x_data = pd.read_csv('HOG2.csv')
    print(raw_data_Y.info())
    #print(raw_data.describe())
    print(raw_data_Y.head(10))
    data = raw_data_Y.copy()
    #Y = data.iloc[1]
    value = []
    for index, row in raw_data_Y.iterrows():
        value.append( row.iloc[0])
    Y = value
    X = x_data
    print(f' Y {Y}')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


    results = dict()

    classif = MLPClassifier(activation='logistic',hidden_layer_sizes=200, max_iter=5000 )
    linear_svc = LinearSVC()
    clfs = [
        # KNeighborsClassifier,
        # SVC,
        # DecisionTreeClassifier,
        # RandomForestClassifier,
        # GaussianNB,
        # #LogisticRegression,
        # GradientBoostingClassifier,
        MLPClassifier,
        # LinearSVC
    ]


    for clf in clfs:
        mdl = Pipeline([
            ('min_max_scaler', MinMaxScaler()),
            ('standard_scaler', StandardScaler()),
            ('classifier', clf())  # RandomForestClassifier(), LinearSVC(), DecisionTreeClassifier()
        ])
        mdl.fit(X_train, y_train)
        results[clf.__name__] = mdl.score(X_test, y_test)

    print(results)



    # with open('wyniki.json', 'w') as output_file:
    #     json.dump(results, output_file, indent=4)

    dump(mdl, 'klasyfikator.pkl')

    stop = time.time()
    print(f'Elapsed time: {stop} seconds')
if __name__ == '__main__':
    main()
