# Importing the few general usecase library
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

## Funtions

# Processing Origin Columns
def process_origin(df):
    df['Origin'] = df['Origin'].map({1 : 'India',
                                    2 : 'USA',
                                    3 : 'Russian'})
    return df

## Tranformer
acceleration_ix, hpower_ix, cylinder_ix = 4, 2, 0
class CustomAttributeAdder(BaseEstimator, TransformerMixin):
    """
    if acc_on_power = True, it return acceraltion per hoursepower and acceration per cylinder 
      else acceration per cyclinder """
    def __init__(self,acc_on_power = True): # no args or kargs        
        self.acc_on_power = acc_on_power
        
    def fit(self,X,y=None):
        return self # nothing else to do
    
    
    def transform(self, X):       
        acc_on_cycle = X[:,acceleration_ix] / X[:,cylinder_ix]       
        if self.acc_on_power:        
            acc_on_power = X[:,acceleration_ix] / X[:,hpower_ix]         
            return np.c_[X,acc_on_power, acc_on_cycle]     
        return np.c_[X, acc_on_cycle]  # Add an column in X at the end 

## Pipeline
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformation
    Argument :
         data : original dataframe
    Return :
         num_attrs : numerical dataframe
         num_pipeline : numerical pipeline object
         '''
    
    numeric = ['float64', 'int64']
    numeric_data = data.select_dtypes(include = numeric)

    # Pipeline for numeric attributes
    ## impute -> adding attribute -> scale them

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribute_adder', CustomAttributeAdder()),
            ('std_scaler', StandardScaler()),
        ])
    return numeric_data, num_pipeline

def pipeline_transfomer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    cat_attrs = ['Origin']
#     print(list(num_attrs))
    # complete pipeline to transform both numerical and cat. attributes
    full_pipline = ColumnTransformer([
        ('num',num_pipeline, list(num_attrs)),
        ('cat',OneHotEncoder(), cat_attrs)
    ])
    prepared_data = full_pipline.fit_transform(data)   
    return prepared_data

## Predict Function
def predict_mpg(config,model):
    '''
    Precition funtion for predicting the miles per gallon(mpg) for a particular vehical config, data
    Aurgument : model,
        model : RandomForest Regressor
        config : Dictinary of vehical configuration
    Return :
          Predicted value of MPG'''
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
        
    preproc_df = process_origin(df)
    prepared_df = pipeline_transfomer(preproc_df)
    y_pred = model.predict(prepared_df)
    
    return y_pred