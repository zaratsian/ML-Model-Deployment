
########################################################################################################
#
#   Sklean Modeling
#
########################################################################################################

import os,sys,csv,re
import time,datetime
import argparse

import pandas as pd
import numpy as np
import scipy as sp
import pickle
import subprocess

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.externals import joblib

########################################################################################################
#
#   Input Data
#
########################################################################################################

# Read local CSV
header = ['Date', 'GameID', 'Drive', 'qtr', 'down', 'time', 'TimeUnder', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'FirstDown', 'posteam', 'DefensiveTeam', 'Yards_Gained', 'Touchdown', 'PlayType', 'PassLength', 'PassLocation', 'RunLocation', 'PosTeamScore', 'DefTeamScore', 'month_day', 'PlayType_lag']
data   = pd.read_csv('{}/data/nfldata2.csv'.format(os.getcwd()), names=header)

########################################################################################################
#
#   Quickly explore data structure
#
########################################################################################################

data.head()
#data.values
data.iloc[0]
data.shape
data.columns
data.dtypes
data.describe()

########################################################################################################
#
#   Model Variables (Specify id, target, numeric variables, and categorical variables)
#
########################################################################################################

var_id              = ''
var_target          = 'Yards_Gained'
var_date            = 'Date'
var_numeric         = ['Drive', 'qtr', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'FirstDown', 'PosTeamScore', 'DefTeamScore', 'month_day', ]
var_category        = ['posteam', 'DefensiveTeam','PlayType','PlayType_lag']

########################################################################################################
#
#   Transformations
#
########################################################################################################

transformed_set             = {}
transformed_set[var_target] = data[var_target]

for var in var_numeric:
    transformed_set[var] = data[var]

'''
for var in var_category:
    transformed_set[var] = data[var].astype('category').cat.codes
'''

category_coding = {}
for var in var_category:
    category_coding[var] = dict( enumerate( data[var].astype('category').cat.categories ))
    transformed_set[var] = data[var].astype('category').cat.codes

extracted_year  = pd.to_datetime(data[var_date]).dt.year
extracted_month = pd.to_datetime(data[var_date]).dt.month
extracted_day   = pd.to_datetime(data[var_date]).dt.day

transformed_set['year']  = extracted_year
transformed_set['month'] = extracted_month
transformed_set['day']   = extracted_day

##########################
# Create transformed DF
##########################

transformed_df = pd.concat([v for k,v in transformed_set.items()], axis=1)
transformed_df.columns = [k for k,v in transformed_set.items()]
transformed_df.head()

########################################################################################################
#
#   Train and Test DFs
#
########################################################################################################

random_number  = pd.DataFrame(np.random.randn(len(transformed_df), 1))
partition_mask = np.random.rand(len(random_number)) <= 0.75

train_data     = transformed_df[partition_mask]
test_data      = transformed_df[~partition_mask]

train_data.shape
test_data.shape

train_inputs   = train_data.drop([var_target], axis=1)
train_target   = train_data[var_target]

test_inputs    = test_data.drop([var_target], axis=1)
test_target    = test_data[var_target]

########################################################################################################
#
#   Gradient Boosting Model (Regression)
#
########################################################################################################

def gb_regressor(train_inputs, train_target, test_inputs, number_of_estimators=100):
    
    model_obj = GradientBoostingRegressor(n_estimators=number_of_estimators, learning_rate=0.1, criterion='friedman_mse', max_depth=3, random_state=12345)
    print('[ INFO ] Training Gradient Boosting Regressor with {} estimators.'.format(number_of_estimators))
    time.sleep(1)
    starttime = datetime.datetime.now()
    model_obj.fit(train_inputs, train_target)
    total_runtime = (datetime.datetime.now() - starttime).seconds
    print('[ INFO ] Total Runtime: {} seconds'.format(total_runtime))
    
    target_predicted = model_obj.predict(test_inputs)
    return model_obj, target_predicted

def evaluate_regression_model(actual, predicted):
    mse      = mean_squared_error(y_true=actual, y_pred=predicted)
    meanae   = mean_absolute_error(y_true=actual, y_pred=predicted)
    medianae = median_absolute_error(y_true=actual, y_pred=predicted)
    r2_score = r2_score(y_true=actual, y_pred=predicted)
    #variance = explained_variance_score(y_true=actual, y_pred=predicted)
    
    print('[ INFO ] Mean Squared Error:     {}'.format(mse))
    print('[ INFO ] Mean Absolute Error:    {}'.format(meanae))
    print('[ INFO ] Median Absolute Error:  {}'.format(medianae))
    print('[ INFO ] R2 Score:               {}'.format(r2_score))

def save_model(model_obj):
    print('[ INFO ] Saving model...')
    model_name  = 'nfl_model'
    bucket_name = 'mlbucket'
    joblib.dump(model_obj, '{}.joblib'.format(model_name))
    print('[ INFO ] Model saved as {}.joblib'.format(model_name))
    subprocess.check_call(['gsutil', 'cp', '{}.joblib'.format(model_name), 'gs://{}/{}'.format(bucket_name, model_name)], stderr=sys.stdout)
    print('[ INFO ] {}.joblib upload to gs://{}/{}'.format(model_name, bucket_name, model_name))



'''
if __name__ == "__main__"
    
    # Arguments (used only for testing)
    args = {"training_data":"", "target_variable"=""}
    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_data",      required=True, type=str, help="Path to local training data")
    ap.add_argument("--target_variable",    required=True, type=str, help="Name of target / label variable")
    args = vars(ap.parse_args())
    
    # Load Dataset
    rawdata = load_rawdata(args['training_data'])
    
    # Transform / Prep dataframe
    transformed_df = transform_df(rawdata)
    
    # Split into train and test dataframes
    traindf, testdf = partition_df(transformed_df)
    
    # Train Model
    model_obj = train_model(traindf, testdf)
    
    # Save Model
    save_model(model_obj)

'''


#ZEND
