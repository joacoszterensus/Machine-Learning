import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def normalizeDataFrame(train, val, binary_features=None):
    """Normaliza solo las columnas numéricas del DataFrame, dejando las categóricas sin cambios"""
    if binary_features is not None:
        num_cols = train.select_dtypes(include=[np.number]).columns.difference(binary_features)
    else:
        num_cols = train.select_dtypes(include=[np.number]).columns

    mean = train[num_cols].mean()  
    std = train[num_cols].std()    

    train_norm = train.copy()
    val_norm = val.copy()

    train_norm[num_cols] = (train[num_cols] - mean) / std  
    val_norm[num_cols] = (val[num_cols] - mean) / std  

    return train_norm, val_norm, mean, std, num_cols

def minMaxScaleDataFrame(train, val, binary_features):
    """Normaliza solo las columnas numéricas del DataFrame usando Min-Max Scaling, dejando las categóricas sin cambios"""

    num_cols = train.select_dtypes(include=[np.number]).columns.difference(binary_features)

    min_values = train[num_cols].min()
    max_values = train[num_cols].max()

    train_scaled = train.copy()
    val_scaled = val.copy()

    train_scaled[num_cols] = (train[num_cols] - min_values) / (max_values - min_values)
    val_scaled[num_cols] = (val[num_cols] - min_values) / (max_values - min_values)

    return train_scaled, val_scaled, min_values, max_values, num_cols


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




def detectar_y_graficar_outliers(df_num, limite=1.5, n_cols=3):
    """
    Detecta outliers con el método IQR y genera una matriz de boxplots para cada variable numérica.
    
    Parámetros:
    - df_num: DataFrame numerico sin el Target.
    - limite: multiplicador de IQR para definir outliers.
    - n_cols: cantidad de columnas en la matriz de plots.
    """
    num_vars = df_num.shape[1]
    n_rows = (num_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(df_num.columns):
        data = df_num[col].dropna()
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower = Q1 - limite * IQR
        upper = Q3 + limite * IQR
        outliers = data[(data < lower) | (data > upper)]

        sns.boxplot(x=data, ax=axes[i], color='skyblue')
        axes[i].set_title(f"{col} ({len(outliers)} outliers)")
        axes[i].grid(True, axis='x', linestyle='--', alpha=0.5)



    plt.tight_layout()
    plt.show()

def remove_outliers_iqr(df, cols, factor=1.5):
    """
    Elimina filas que contengan outliers (según IQR) en cualquiera de las columnas dadas.
    Conserva los valores NaN originales.

    Parámetros:
    - df: DataFrame de entrada
    - cols: lista de columnas numéricas donde buscar outliers
    - factor: multiplicador del IQR para definir el umbral (por defecto 1.5)

    Devuelve:
    - df sin filas con outliers en las columnas seleccionadas, manteniendo NaNs
    """

    mask = np.ones(len(df), dtype=bool)

    for col in cols:
        q1 = np.nanpercentile(df[col], 25)
        q3 = np.nanpercentile(df[col], 75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        col_mask = ((df[col] >= lower_bound) & (df[col] <= upper_bound)) | df[col].isna()
        mask &= col_mask

    return df[mask]