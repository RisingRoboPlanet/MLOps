# library doc string
'''
Module to determine customers who have high probability to churn
Author: "Diva Alwi"
Date: 29 June 2022
'''

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import logging
import warnings
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

warnings.filterwarnings("ignore")

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # Load dataframe
    dataset = pd.read_csv(pth)

    # Convert attribution flag to numeric values
    def lambda_functions(val): return 0 if val == "Existing Customer" else 1
    dataset['Churn'] = dataset['Attrition_Flag'].apply(lambda_functions)

    return dataset


def perform_eda(dataset):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create destination path for images, if not exists
    eda_pth = "./images/eda/"
    if not os.path.exists(eda_pth):
        os.makedirs(eda_pth)

    # List of columns to plot
    plot_columns = ["Churn", "Customer_Age", "Total_Trans_Ct", "Heatmap"]

    for column in plot_columns:
        plt.figure(figsize=(20, 10))

        # Plot churn distribution
        if column == 'Churn':
            dataset['Churn'].hist()
            plt.title("churn distribution")

        # Plot customer ages distribution
        elif column == 'Customer_Age':
            dataset['Customer_Age'].hist()
            plt.title("customer age distribution")

        # Plot marital status distribution
        elif column == 'Marital_Status':
            dataset.Marital_Status.\
                value_counts('normalize').\
                plot(kind='bar')
            plt.title("marital status distribution")

        # Plot total trans ct distribution
        elif column == 'Total_Trans_Ct':
            sns.distplot(dataset['Total_Trans_Ct'])
            plt.title("total trans ct distribution")

        # Plot Heatmap
        elif column == 'Heatmap':
            sns.heatmap(dataset.corr(),
                        annot=False,
                        cmap='Dark2_r',
                        linewidths=2)
            plt.title("heatmap")

        plt.savefig(os.path.join(eda_pth, "{}.jpg".format(column)))
        plt.close()


def encoder_helper(dataset, category_lst, response):
    '''
    helper function to turn each categorical column into a new column
    with propotion of churn for each category - associated with cell 015
    from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                      could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for columns in category_lst:
        lst = []
        group = dataset.groupby(columns).mean()[response]

        for val in dataset[columns]:
            lst.append(group.loc[val])

        dataset[columns + '_' + response] = lst

    return dataset


def perform_feature_engineering(dataset, response):
    '''
    input:
            df: pandas dataframe
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    # Categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Keep columns
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y_label = dataset['Churn']
    X_dataset = pd.DataFrame()

    # Encode categorical columns
    dataset = encoder_helper(dataset, cat_columns, response)
    X_dataset[keep_cols] = dataset[keep_cols]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_label, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    # Random Forest
    plt.figure()
    plt.rc('figure', figsize=(20, 10))
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6,
             'Random Forest Test',
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.jpg')
    plt.close()

    # Logistic Regression
    plt.figure()
    plt.rc('figure', figsize=(20, 10))
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6,
             'Logistic Regression Test',
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.jpg')
    plt.close()


def feature_importance_plot(model, X_dataset, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    # Feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    names = [X_dataset.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_dataset.shape[1]), importances[indices])
    plt.xticks(range(X_dataset.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    '''
    # Model init
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # Grid search - fitting : Random Forest Classifier
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic regression fitting
    lrc.fit(X_train, y_train)

    # Test prediction : Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Test prediction : Logistic Regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Roc curve plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_,
                       X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.jpg')
    plt.close()

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # store feature importances plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train,
                            './images/results/feature_importances.jpg')


if __name__ == "__main__":
    BANK_DATA = import_data("./data/bank_data.csv")

    perform_eda(BANK_DATA)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        BANK_DATA,
        'Churn')

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
