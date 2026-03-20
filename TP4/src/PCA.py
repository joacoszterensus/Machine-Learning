import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    Implementación de PCA usando solo NumPy.
    Métodos:
     - fit(X): calcula media y vectores principales.
     - transform(X, k): proyecta X en k componentes.
     - inverse_transform(scores): reconstruye datos desde puntuaciones.
     - reconstruction_error(X, k): MSE de reconstrucción con k componentes.
     - plot_reconstruction_error(X, max_k): grafica MSE vs número de componentes hasta max_k.
    """
    def __init__(self):
        self.mean_ = None
        self.components_ = None  # Vt de SVD

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt
        self.singular_values_ = S  # <-- guardamos S
        return self

    def transform(self, X, k):
        """
        Proyecta X centrado en los primeros k componentes.
        Devuelve puntuaciones de forma (n_samples, k).
        """
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        return Xc.dot(self.components_[:k].T)

    def inverse_transform(self, scores):
        """
        Reconstruye datos desde puntuaciones: scores.dot(components) + mean.
        """
        return scores.dot(self.components_[:scores.shape[1]]) + self.mean_

    def reconstruction_error(self, X, k):
        """
        Calcula el MSE de reconstrucción usando k componentes.
        """
        scores = self.transform(X, k)
        X_recon = self.inverse_transform(scores)
        return np.mean((X - X_recon) ** 2)

    def plot_reconstruction_error(self, X, max_k=100):
        """
        Grafica el MSE de reconstrucción para k=1..max_k.
        """
        ks = np.arange(1, max_k + 1)
        mses = [self.reconstruction_error(X, k) for k in ks]

        plt.figure(figsize=(8,5))
        plt.plot(ks, mses, 'o-')
        plt.xlabel('Número de componentes principales')
        plt.ylabel('Error cuadrático medio de reconstrucción')
        plt.title('PCAMSE vs #componentes')
        plt.grid(True)
        plt.show()

    def plot_singular_values(self):
        """
        Grafica los valores singulares ordenados.
        """
        if self.singular_values_ is None:
            raise ValueError("Primero hay que ajustar el modelo con fit(X)")
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(1, len(self.singular_values_)+1), self.singular_values_, 'o-')
        plt.xlabel('Índice de componente principal')
        plt.ylabel('Valor singular')
        plt.title('Valores singulares del conjunto de datos')
        plt.grid(True)
        plt.show()
