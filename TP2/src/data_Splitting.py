
import numpy as np

def divideDataFrame(df:object, ratio:int):
    """divide el dataframe en dos partes, una para entrenamiento con cierto ratio y otra para test con lo restante"""
    train = df.sample(frac=ratio, random_state=42)
    val = df.drop(train.index)
    return train, val

def cross_validation(model, X, y, lambda_values,metric, k=5):
    """Calcula el error cuadrático medio (ECM) promedio en validación cruzada para un modelo de regresión."""
    n = X.shape[0]
    fold_size = n // k  
    indices = np.arange(n)
    np.random.shuffle(indices)  

    values = []

    for lambda_reg in lambda_values:
        folds = []
        model.set_lambda(lambda_reg)
        for i in range(k):
            val_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.delete(indices, np.arange(i * fold_size, (i + 1) * fold_size))
            
            X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]


            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)

            error =metric(y_val, y_val_pred)
            folds.append(error)

        values.append(np.mean(folds))

    return np.array(values)