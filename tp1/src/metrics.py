
import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    """ Calcula el Error Cuadrático Medio (Mean Squared Error). """
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    """Calcula el Error Absoluto Medio (Mean Absolute Error)."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio (Root Mean Squared Error)."""
    return np.sqrt(mse(y_true, y_pred))

def errors(true,pred):
    """printea el Error Cuadrático Medio (Mean Squared Error), 
    el Error Absoluto Medio (Mean Absolute Error) y la Raíz del Error Cuadrático Medio (Root Mean Squared Error)."""
    print("MSE: ",mse(true,pred))
    print("MAE: ",mae(true,pred))
    print("RMSE: ",rmse(true,pred))

def plot_error_histogram(Y_true, Y_pred, bins=30):
    """
    Genera un histograma de los errores entre los valores reales y las predicciones.
    """
    errors = Y_true - Y_pred

    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(x=0, color="red", linestyle="dashed", linewidth=2, label="Error = 0")

    plt.xlabel("Error (Real - Predicho)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma de Errores del Modelo")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.show()
