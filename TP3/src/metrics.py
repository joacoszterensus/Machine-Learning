import numpy as np

def cross_entropy_loss( y_true, y_pred,weights, l2_lambda=None):
    """
    Calcula la función de pérdida de entropía cruzada.
    y_true: etiquetas verdaderas (one-hot encoded)
    y_pred: predicciones del modelo (probabilidades)
    """
   
    m = y_true.shape[0]
    log_likelihood = np.log(y_pred[range(m), y_true])
    loss = -np.sum(log_likelihood) / m
    if l2_lambda:
        l2_term = sum(np.sum(W**2) for W in weights)
        loss += (l2_lambda / (2 * m)) * l2_term
    return loss

# 1) Accuracy
def accuracy(y_true, y_pred):
    """
    calcula la precisión de las predicciones
    :param y_true: etiquetas verdaderas
    :param y_pred: etiquetas predichas
    :return: precisión
    """

    return np.mean(y_true == y_pred)


