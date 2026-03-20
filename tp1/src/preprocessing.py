import pandas as pd
import numpy as np

def normalizeDataFrame(train: pd.DataFrame, val: pd.DataFrame, binary_features: list):
    """Normaliza solo las columnas numéricas del DataFrame, dejando las categóricas sin cambios"""

    num_cols = train.select_dtypes(include=[np.number]).columns.difference(binary_features)

    mean = train[num_cols].mean()  
    std = train[num_cols].std()    

    train_norm = train.copy()
    val_norm = val.copy()

    train_norm[num_cols] = (train[num_cols] - mean) / std  
    val_norm[num_cols] = (val[num_cols] - mean) / std  

    return train_norm, val_norm, mean, std, num_cols

def fillNaNs(df:object):
    """rellena los valores NaNs con la mediana de la columna"""
    return df.fillna(df.median())

def one_hot_encoding(df, columns):
    """Aplica One-Hot Encoding a las columnas especificadas del DataFrame."""
    df_encoded = df.copy()
    
    for col in columns:
        unique_vals = df_encoded[col].unique()
        for val in unique_vals:
            new_col = f"{col}_{val}"
            df_encoded[new_col] = (df_encoded[col] == val).astype(int)
        df_encoded = df_encoded.drop(columns=[col])
        
    return df_encoded

