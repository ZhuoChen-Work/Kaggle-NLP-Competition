import pandas as pd
import numpy as np

def extract_data(path_in,path_out,col):
    """extract data for training, remove the useless cols"""
    data_df = pd.read_csv(path_in)
    data_df = data_df[col].fillna(0)
    data_df.to_csv(path_out,index=False)
    
def split_data(data_df,validate_col=['comment_text','target']):
    """split the data into train_df and validate_df"""
    n = len(data_df)
    train_df = data_df[:int(0.9*n)]
    # validate_df only need comment_text and ground truth targets
    validate_df = data_df[int(0.9*n):][validate_col]
    return train_df,validate_df

def adapt_weight(df,identity_columns,spe_columns):
    """set weights for different type training sambles"""
    
    #The based weight
    weights = np.ones(len(df)) / 4

    # Subgroup  positive  
    temp_index = (df[identity_columns].values>=0.5).max(axis=1)
    weights[temp_index] += 0.25

    # Background Positive, Subgroup Negative
    temp_index = (df['target'].values>=0.5) * ((df[identity_columns].values<0.5).sum(axis=1) == len(identity_columns))
    weights[temp_index] += 0.25

    # Background Negative, Subgroup Positive
    temp_index = (df['target'].values<0.5) * (df[identity_columns].values>=0.5).max(axis=1)
    weights[temp_index] += 0.25

    # Background Positive, special-Subgroup Negative
    temp_index = (df['target'].values>=0.5) * ((df[spe_columns].values<0.5).sum(axis=1) == len(spe_columns))
    weights[temp_index] += 0.125

    # Background Positive, special-Subgroup Positive  
    temp_index = (df['target'].values<0.5) * (df[spe_columns].values>=0.5).max(axis=1)
    weights[temp_index] += 0.125

    return weights