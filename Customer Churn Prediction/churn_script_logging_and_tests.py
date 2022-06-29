'''
Perfom test & logging for churn_library.py file

Author: "Diva Alwi"
Date: 29 June 2022
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist
    with the other test functions
    '''
    try:
        dataset = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataset.shape[0] > 0
        assert dataset.shape[1] > 0
        return dataset
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to\
        	have rows and columns")
        raise err

def test_eda(perform_eda, dataset):
    '''
    test perform eda function
    
    '''
    perform_eda(dataset)
    path = "./images/eda"
    
    # Checking images results
    try:
        dir_img = os.listdir(path)
        assert len(dir_img) >= 4
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: FAIT, some images aren\'t saved.")
        raise err


def test_encoder_helper(encoder_helper, dataset):
    '''
    test encoder helper
    '''
    # Checking if cat_columns are available in dataset
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]

    dataset = encoder_helper(dataset, cat_columns, 'Churn')

    try:
        for element in cat_columns:
            assert element in dataset.columns
        logging.info("Testing encoder_helper: SUCCESS")
        return dataset
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: FAIL, there are missing colmuns.")
        return err

def test_perform_feature_engineering(perform_feature_engineering, dataset):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(
    dataset, 'Churn')
    
    try:
        # check shape and length
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAIL, output missing.")
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    
    path = "./images/results/"
    try:
        dir_img = os.listdir(path)
        assert len(dir_img) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: "
        	"FAIL, all images aren\'t saved")
        raise err
    
    path = "./models/"
    try:
        dir_model = os.listdir(path)
        assert len(dir_model) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: "
        	"FAIL, all models aren\'t saved")
        raise err

if __name__ == "__main__":
    DATA = test_import(cl.import_data)
    test_eda(cl.perform_eda, DATA)
    DATA = test_encoder_helper(cl.encoder_helper, DATA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, DATA)
    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)








