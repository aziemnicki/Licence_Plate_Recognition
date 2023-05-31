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
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from joblib import dump

def main():

    raw_data_Y = pd.read_csv('dane2.csv', header=None)
    x_data = pd.read_csv('data.csv', header=None)

    # df = pd.read_csv('polaczone.csv', header=None)

    # print(raw_data_Y.info())
    # print(x_data.describe())
    # print(raw_data_Y.head(10))
    value = []
    for index, row in raw_data_Y.iterrows():
        value.append(row.iloc[0])
    Y = np.ravel(raw_data_Y)
    # print(f' Y {Y}')

    X = x_data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


    results = dict()

    classif = MLPClassifier(activation='logistic',hidden_layer_sizes=200, max_iter=50000 )
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
            ('classifier', classif)  # RandomForestClassifier(), LinearSVC(), DecisionTreeClassifier()
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
