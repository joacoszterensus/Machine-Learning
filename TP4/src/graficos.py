import numpy as np
import matplotlib.pyplot as plt

from .Kmeans import KMeans
from .GMM import GMM



def graficar_ganancias_decrecientes(X, k_max=10, random_state=42):
    """Grafica la inercia (L) en función de la cantidad de clusters K"""
    inertias = []
    ks = range(1, k_max + 1)

    for k in ks:
        modelo = KMeans(n_clusters=k, random_state=random_state)
        modelo.fit(X)
        inertias.append(modelo.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, 'bo-')
    plt.xlabel("Cantidad de clusters K")
    plt.ylabel("Inercia total L")
    plt.title("Método de ganancias decrecientes (Elbow)")
    plt.grid(True)
    plt.show()

    return inertias

def graficar_clusters(X, labels, centroids, title="Clustering"):
    """Grafica los datos X coloreados según su cluster y los centroides"""
    plt.figure(figsize=(8, 6))
    n_clusters = len(np.unique(labels))

    for i in range(n_clusters):
        puntos = X[labels == i]
        plt.scatter(puntos[:, 0], puntos[:, 1], s=30, label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroides')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()

def ejecutar_kmeans_completo(X, k_max=10, k_final=None, random_state=0):
    """
    Ejecuta el análisis completo de K-means:
    1. Grafica L vs K 
    2. Ajusta con K óptimo y grafica los clusters
    """
    graficar_ganancias_decrecientes(X, k_max=k_max, random_state=random_state)

    print(f"Paso 2: Ejecutando K-means con K = {k_final}...")
    modelo = KMeans(n_clusters=k_final, random_state=random_state)
    modelo.fit(X)
    graficar_clusters(X, modelo.labels_, modelo.centroids_, title=f"K-means con K = {k_final}")



def graficar_log_verosimilitud(X, k_max=10, random_state=42):
    """Grafica la log-verosimilitud en función de la cantidad de componentes K"""
    log_liks = []
    ks = range(1, k_max + 1)

    for k in ks:
        modelo = GMM(n_components=k, random_state=random_state)
        modelo.fit(X)
        # puntuación total de muestras: log prob de X bajo el modelo
        log_liks.append(modelo.score(X) * X.shape[0])

    plt.figure(figsize=(8, 5))
    plt.plot(ks, log_liks, 'bo-')
    plt.xlabel("Cantidad de componentes K")
    plt.ylabel("Log-verosimilitud total")
    plt.title("Método de ganancias decrecientes (GMM Elbow)")
    plt.grid(True)
    plt.show()

    return log_liks

def graficar_clusters_gmm(X, labels, means, title="GMM Clustering"):
    """Grafica los datos X coloreados según su cluster y los centros de las gaussianas"""
    plt.figure(figsize=(8, 6))
    n_clusters = means.shape[0]

    for i in range(n_clusters):
        puntos = X[labels == i]
        plt.scatter(puntos[:, 0], puntos[:, 1], s=30)

    plt.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=200)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.show()

def ejecutar_gmm_completo(X, k_max=10, k_final=None, random_state=0):
    """
    Ejecuta el análisis completo de GMM:
    1. Grafica log-verosimilitud vs K
    2. Ajusta con K óptimo y grafica los clusters
    """
    # Paso 1
    log_liks = graficar_log_verosimilitud(X, k_max=k_max, random_state=random_state)

    # Paso 2
    if k_final is None:
        raise ValueError("Debe especificar k_final con el número de componentes deseado")
    print(f"Paso 2: Ejecutando GMM con K = {k_final}...")
    modelo = GMM(n_components=k_final, max_iter=100)
    modelo.fit(X)
    labels = modelo.predict(X)
    means = modelo.means_

    graficar_clusters_gmm(X, labels, means,
                          title=f"GMM con K = {k_final}")

    return log_liks, labels, means