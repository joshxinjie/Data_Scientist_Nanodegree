# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import math
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

def encode_person_id(profile_df, transcript_df):
    '''
    DESCRIPTION:
    Label encode hash values of id column for profile dataframe and person column for 
    transcript dataframe. Generates per_id variable.
    
    INPUTS:
    profile_df - the profile dataframe
    transcript_df - the transcript dataframe
    
    OUTPUTS:
    profile_df - The profile dataframe with the newly encoded per_id
    transcript_df - The transcript dataframe with the newly encoded per_id
    person_encoder - The label encoder object used to encode the hash values.
                     Can be used to retrieve original encodings
    '''
    person_encoder = LabelEncoder()
    profile_df['per_id'] = person_encoder.fit_transform(profile_df['id'])
    transcript_df['per_id'] = person_encoder.transform(transcript_df['person'])
    
    profile_df.drop(['id'], axis=1, inplace=True)
    transcript_df.drop(['person'], axis=1, inplace=True)
    
    return profile_df, transcript_df, person_encoder

def encode_offer_id(portfolio_df, transcript_offer_df):
    '''
    DESCRIPTION:
    Encode hash values of id column for portfolio dataframe and value_id_amt column for 
    transcript_offer dataframe. Generates offer_id.
    
    INPUTS:
    profile_df - the profile dataframe
    transcript_df - the transcript dataframe
    
    OUTPUTS:
    profile_df - The profile dataframe with the newly encoded offer_id
    transcript_df - The transcript dataframe with the newly encoded offer_id
    offer_encoder - The label encoder object used to encode the hash values.
                     Can be used to retrieve original encodings
    '''
    offer_encoder = LabelEncoder()
    portfolio_df['offer_id'] = offer_encoder.fit_transform(portfolio_df['id'])
    transcript_offer_df['offer_id'] = offer_encoder.transform(transcript_offer_df['value_id_amt'])
    
    portfolio_df.drop(['id'], axis=1, inplace=True)
    transcript_offer_df.drop(['value_type', 'value_id_amt'], axis=1, inplace=True)
    
    return portfolio_df, transcript_offer_df, offer_encoder


def convert_data_type(df, int_var_list=None, float_var_list=None, str_var_list=None):
    '''
    DESCRIPTION:
    Ensure data types are consistent. Convert values in dataframe 
    into desired data types.
    
    INPUTS:
    df - The dataframe that we want to work with
    int_var_list - The list of variable to be converted to integers
    float_var_list - The list of variables to be converted to floats
    str_var_list - The list of variables to be converted to strings
    
    OUTPUTS:
    df - The dataframe that with the variables converted to the desired types
    '''
    if int_var_list != None:
        for int_var in int_var_list:
            df[int_var] = df[int_var].apply(lambda x: int(x))
        
    if float_var_list != None:
        for float_var in float_var_list:
            df[float_var] = df[float_var].apply(lambda x: float(x))
        
    if str_var_list != None:
        for str_var in str_var_list:
            df[str_var] = df[str_var].apply(lambda x: str(x))
        
    return df

def encode_channel(portfolio_df):
    '''
    DESCRIPTION:
    Extract and perform one-hot encodings for the available channels in portfolio
    
    INPUTS:
    portfolio_df - The portfolio dataframe
    
    OUTPUTS:
    portfolio_df - The portfolio dataframe with the one-hot encoded channels
    '''
    # find all available channels
    available_channels = []
    for channel_list in portfolio_df['channels']:
        for channel in channel_list:
            if channel not in available_channels:
                available_channels.append(channel)
    
    # one-hot encode channel
    for channel in available_channels:
        portfolio_df[channel] = portfolio_df['channels'].apply(lambda x: 1 if channel in x else 0)
    
    # drop channels column
    portfolio_df.drop(['channels'], axis=1, inplace=True)
    
    return portfolio_df