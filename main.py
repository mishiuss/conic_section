import pandas as pd
from Symbolic_regression_classification_generator import gen_classification_symbolic
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
import argparse
import pickle
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

def data_generation(n_train=1000, n_test=300):
    print('Running train data generation ...')
    train_gen = gen_classification_symbolic(m='x2^2-4*x1*x3', n_samples=n_train)
    train_df = pd.DataFrame(train_gen, columns=['x' + str(i) for i in range(1, 4)] + ['y'])
    train_df.to_csv('train.csv', index=False)
    print('train: ', train_df.shape)

    print('Running test data generation ...')
    test_gen = gen_classification_symbolic(m='x2^2-4*x1*x3', n_samples=n_test)
    test_df = pd.DataFrame(test_gen, columns=['x' + str(i) for i in range(1, 4)] + ['y'])
    test_df.to_csv('test.csv', index=False)
    print('test: ', test_df.shape)

def read_data(file_train, file_test):
    DIR = os.getcwd()
    train = pd.read_csv(DIR + file_train)
    test = pd.read_csv(DIR + file_test)
    y = train['y']
    X = train.drop('y', axis=1)
    y_test = test['y']
    X_test = test.drop('y', axis=1)
    return X, y, X_test, y_test

def train_model(X, y, search_space, name, scoring='accuracy', cv=3):
    print('Running GridSearchCV  ...')
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    clf = GridSearchCV(pipe, search_space, scoring=scoring, cv=cv)
    best_model = clf.fit(X, y)
    print("Best parameters set found on train set:")
    print(best_model.best_params_)
    print("Save the model to disk: ", name)
    pickle.dump(best_model, open(name, 'wb'))
    return best_model

def test_model(best_model, X_test, y_test):
    result = best_model.score(X_test, y_test)
    print("\nThe score computed on the full test set:")
    print(result)
    y_true, y_pred = y_test, best_model.predict(X_test)
    report = classification_report(y_true, y_pred)
    print("\nDetailed classification report:")
    print(report)
    with open('testres.txt', 'a') as the_file:
        the_file.write("Detailed classification report:\n")
        the_file.write("The scores are computed on the full test set:\n")
        the_file.write(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_gen', default=False)
    parser.add_argument('--train', default=False)
    parser.add_argument('--model_name', default='finalized_model.pkl')
    parser.add_argument('--test', default=True)
    parser.add_argument('--ABC', '--names-list', nargs="*", default=[1, 0, 0])
    parser.add_argument('--vis', default=True)
    args = parser.parse_args()
    if args.data_gen:
        data_generation(n_train=5000, n_test=500)

    X, y, X_test, y_test = read_data('/train.csv', '/test.csv')

    params = [
        {
         'classifier': [SVC()],
         'classifier__kernel': ['linear', 'rbf'],
         'classifier__C': [1, 10]
        },
        {
          'classifier': [RandomForestClassifier()],
          'classifier__n_estimators': [10, 100, 1000],
          'classifier__max_features': [1, 2, 3]
        }
    ]

    if args.train:
        best_model = train_model(X, y, search_space=params, scoring="f1_weighted",
                                 name=args.model_name, cv=5)

    else:
        print("\nLoad the model from disk: ", args.model_name)
        best_model = pickle.load(open(args.model_name, 'rb'))
        print("Parameters of loaded model:")
        print(best_model.best_params_)

    with open('testres.txt', 'w') as the_file:
        the_file.write("Parameters of model:\n")
        for k in sorted(best_model.best_params_.keys()):
            the_file.write("'%s':'%s', \n" % (k, best_model.best_params_[k]))

    if args.test:
        test_model(best_model, X_test, y_test)

    if args.ABC is not None:
        print('\nInput A, B, C:', args.ABC)
        my = np.asarray(args.ABC).reshape(1, -1)
        if len(my[np.abs(my)>1]) is not 0 :
            sys.exit("\nError: Absolute values of coefficients must be <= 1")
        print("Predicted conic section:", best_model.predict(my))

    if args.vis:
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        x, y = np.meshgrid(x, y)
        a, b, c, d, e, f = my[0, 0], my[0, 1], my[0, 2], 0, 1, 1
        plt.contour(x, y, (a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f), [0])
        plt.show()

if __name__ == "__main__":
    main()
