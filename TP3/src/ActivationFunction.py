import numpy as np

def relu( Z):
    """
    Z: matriz de activaciones
    devuelve: matriz de activaciones después de aplicar la función ReLU
    """
    return np.maximum(0, Z)

def relu_derivative( Z):
    """
    Z: matriz de activaciones
    devuelve: matriz de derivadas de la función ReLU
    """

    return (Z > 0).astype(float)

def softmax( Z):
    """
    Z: matriz de activaciones
    devuelve: matriz de probabilidades después de aplicar la función softmax
    """
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def softmax_derivative(logits, y_onehot):
    """
    logits: salidas de la última capa antes de softmax, shape (batch_size, n_classes)
    y_onehot: etiquetas one-hot, same shape
    devuelve: dZ = dL/dz
    """
    a = softmax(logits)    # (batch_size, n_classes)
    return a - y_onehot    # derivada conjunta softmax + cross-entropy