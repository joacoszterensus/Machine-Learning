import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=150, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def initialize_centroids(self, X):
        """Inicializa usando k-means++ para mejores centroides iniciales"""
        n_samples, _ = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)

        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        # primer centroide aleatorio
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]

        # k-means++
        dist = np.full(n_samples, np.inf)
        for i in range(1, self.n_clusters):
            # distancia mínima al conjunto de centroides actuales
            dist = np.minimum(dist, np.sum((X - centroids[i-1])**2, axis=1))
            probs = dist / dist.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = X[next_idx]
        return centroids

    def assign_clusters(self, X, centroids):
        """Vectoriza el cálculo de etiquetas"""
        # distancias de cada punto a cada centroide
        dist = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(dist, axis=1)        # asigna cada muestra al centroide más cercano


    def update_centroids(self, X, labels):
        """Vectoriza la actualización de centroides"""
        new_centroids = np.zeros_like(self.centroids_)
        for i in range(self.n_clusters):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                # evitar cluster vacío: reubicar aleatoriamente
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
        return new_centroids

    def compute_inertia(self, X, labels, centroids):
        """Inercia vectorizada"""
        #  distancia al cuadrado
        dist = np.linalg.norm(X - centroids[labels], axis=1)**2
        return dist.sum()

    def fit(self, X):
        self.centroids_ = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = self.assign_clusters(X, self.centroids_)
            new_centroids = self.update_centroids(X, labels)
            # comprobar convergencia en movimiento de centroides
            if np.allclose(self.centroids_, new_centroids, atol=self.tol):
                break
            self.centroids_ = new_centroids
        self.labels_ = self.assign_clusters(X, self.centroids_)
        self.inertia_ = self.compute_inertia(X, self.labels_, self.centroids_)
        return self

    def predict(self, X):
        return self.assign_clusters(X, self.centroids_)


