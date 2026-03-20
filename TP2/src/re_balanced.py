import numpy as np
import pandas as pd


def undersampling(X, y, random_state=42):
    """
    Random undersampling binario para balancear clases.
    Devuelve un DataFrame con X e y concatenados.
    """
    np.random.seed(random_state)

    clases = np.unique(y)
    if len(clases) != 2:
        raise ValueError("Esta función solo soporta problemas binarios.")

    counts = [(y == c).sum() for c in clases]
    clase_mayoritaria = clases[np.argmax(counts)]
    clase_minoritaria = clases[np.argmin(counts)]

    idx_mayoritaria = np.where(y == clase_mayoritaria)[0]
    idx_minoritaria = np.where(y == clase_minoritaria)[0]

    idx_mayoritaria_down = np.random.choice(idx_mayoritaria, size=len(idx_minoritaria), replace=False)
    idx_final = np.concatenate([idx_minoritaria, idx_mayoritaria_down])
    np.random.shuffle(idx_final)

    df = X.copy()
    df['Diagnosis'] = y.values

    df_final = df.iloc[idx_final].reset_index(drop=True)

    return df_final



def oversampling_duplicacion(X, y, random_state=42):
    """
    Oversampling por duplicación para balancear clases.
    Se selecciona aleatoriamente una muestra de la clase minoritaria
    para igualar el número de muestras de la clase mayoritaria.
    
    Parámetros:
    - X: DataFrame con las features.
    - y: Serie o array con la variable objetivo.
    - random_state: semilla para reproducibilidad.
    
    Devuelve:
    - DataFrame resultante con las features y la variable objetivo balanceada.
    """
    np.random.seed(random_state)

    # Asegurarse de trabajar con un array
    clases = np.unique(y)
    if len(clases) != 2:
        raise ValueError("La función solo está implementada para problemas binarios.")
    
    counts = [(y == c).sum() for c in clases]
    clase_mayoritaria = clases[np.argmax(counts)]
    clase_minoritaria = clases[np.argmin(counts)]

    idx_mayoritaria = np.where(y == clase_mayoritaria)[0]
    idx_minoritaria = np.where(y == clase_minoritaria)[0]

    idx_minoritaria_up = np.random.choice(idx_minoritaria, size=len(idx_mayoritaria), replace=True)

    idx_final = np.concatenate([idx_mayoritaria, idx_minoritaria_up])
    np.random.shuffle(idx_final)

    df = X.copy()
    df['Diagnosis'] = y.values

    df_final = df.iloc[idx_final].reset_index(drop=True)

    return df_final


def oversampling_smote(X, y, k=5, random_state=42):
    """
    Oversampling usando SMOTE (Synthetic Minority Over-sampling Technique).
    Genera muestras sintéticas para la clase minoritaria basándose en sus vecinos más cercanos.
    
    Parámetros:
    - X: DataFrame o array de características.
    - y: Series o array de etiquetas.
    - k: número de vecinos a considerar.
    - random_state: semilla para la reproducibilidad.
    
    Retorna:
    - df: DataFrame con las muestras originales y las sintéticas.
    
    Nota: La función asume un problema de clasificación binaria.
    """
    np.random.seed(random_state)
    
    X_array = X.values
    y_array = y.values
       
    clases = np.unique(y_array)
    counts = np.array([(y_array == c).sum() for c in clases])
    clase_mayoritaria = clases[np.argmax(counts)]
    clase_minoritaria = clases[np.argmin(counts)]

    X_min = X_array[y_array == clase_minoritaria]
    X_maj = X_array[y_array == clase_mayoritaria]


    n_samples_needed = len(X_maj) - len(X_min)
    synthetic_samples = []

    for _ in range(n_samples_needed):
        i = np.random.randint(0, len(X_min))
        xi = X_min[i]
        
        distancias = np.linalg.norm(X_min - xi, axis=1)
        vecinos_idx = np.argsort(distancias)[1:k+1]
        
        v = X_min[np.random.choice(vecinos_idx)]
        
        diff = v - xi
        gap = np.random.rand()
        synthetic = xi + gap * diff
        synthetic_samples.append(synthetic)

    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(len(X_synthetic), clase_minoritaria)

    X_final = np.vstack((X_array, X_synthetic))
    y_final = np.concatenate((y_array, y_synthetic))

    X_final = pd.DataFrame(X_final, columns=X.columns)
    y_final = pd.Series(y_final, name=y.name)

    df = pd.concat([X_final, y_final], axis=1)
    return df

def calcular_pesos(y):
    """
    Calcula los pesos para cada clase en función de su frecuencia.
    Para cada clase, el peso se define como: (frecuencia de la clase mayoritaria) / (frecuencia de la clase).
    
    Parámetros:
    - y: array o Serie con las etiquetas.
    
    Retorna:
    - pesos: diccionario con la clave la clase y el valor su peso.
    """
    clases = np.unique(y)
    counts = {c: (y == c).sum() for c in clases}
    max_count = max(counts.values())
    pesos = {c: max_count / counts[c] for c in clases}
    return pesos

def cost_reweighting(X, Y):
    """
    Simula el cost re-weighting replicando cada muestra de X, Y de acuerdo al peso de su clase.
    
    Parámetros:
    - X: DataFrame de shape (n_samples, n_features).
    - Y: Serie de shape (n_samples,).
    
    Retorna:
    - df: DataFrame que contiene las muestras replicadas y sus etiquetas.
    
    Nota: La función asume que X es un DataFrame y Y una Serie.
    """
    Y_arr = Y.iloc[:].values  
    pesos = calcular_pesos(Y_arr)
    
    X_rep = []
    Y_rep = []
    
    for i in range(len(Y_arr)):
        w = pesos[Y_arr[i]]
        repeticiones = int(np.floor(w))
        parte_fraccionaria = w - repeticiones
        
        if np.random.rand() < parte_fraccionaria:
            repeticiones += 1
        
        for _ in range(repeticiones):
            X_rep.append(X.iloc[i].values)
            Y_rep.append(Y_arr[i])
    
    X_rep = pd.DataFrame(np.array(X_rep), columns=X.columns)
    Y_rep = pd.Series(Y_rep, name=Y.name)
    
    df = pd.concat([X_rep, Y_rep], axis=1)
    return df
