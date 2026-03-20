
import numpy as np

def divideDataFrame(df:object, ratio:int):
    """divide el dataframe en dos partes, una para entrenamiento con cierto ratio y otra para test con lo restante"""
    train = df.sample(frac=ratio, random_state=42)
    val = df.drop(train.index)
    return train, val

def cross_validation_mse(model, X, y, lambda_values, k=5):
    """Calcula el error cuadrático medio (ECM) promedio en validación cruzada para un modelo de regresión."""
    n = X.shape[0]
    fold_size = n // k  
    indices = np.arange(n)
    np.random.shuffle(indices)  

    ecm_values = []

    for lambda_reg in lambda_values:
        ecm_folds = []

        for i in range(k):
            val_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.delete(indices, np.arange(i * fold_size, (i + 1) * fold_size))
            
            X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            model.gradientDescentTraining(X_train, y_train, L2=lambda_reg)

            y_val_pred = model.predictGradientDescent(X_val)

            error = np.mean((y_val_pred - y_val) ** 2)
            ecm_folds.append(error)

        ecm_values.append(np.mean(ecm_folds))

    return np.array(ecm_values)