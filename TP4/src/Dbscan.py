import numpy as np

class DBSCAN:
    """
    Parámetros:
    - eps: radio de vecindad
    - min_samples: número mínimo de muestras para formar un cluster
    """
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        """
        Ajusta el modelo DBSCAN a los datos X.
        X: array de forma (n_samples, n_features)
        Después de ejecutar fit, los labels están en self.labels_.
        """
        n_samples = X.shape[0]
        # 0: no visitado, -1: ruido, 1,2,...: cluster IDs
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != 0:
                continue  # ya visitado

            # Encontrar vecinos
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # marcar como ruido
            else:
                cluster_id += 1
                self._expand_cluster(X, labels, i, neighbors, cluster_id)

        self.labels_ = labels
        return self

    def _region_query(self, X, idx):
        """
        Devuelve los índices de puntos en X que están dentro de eps de X[idx].
        """
        diff = X - X[idx]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        neighbors = np.where(dist <= self.eps)[0]
        return neighbors.tolist()

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id):
        """
        Expande el cluster con ID cluster_id empezando desde el punto idx.
        """
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            j = neighbors[i]
            if labels[j] == -1:
                labels[j] = cluster_id
            elif labels[j] == 0:
                labels[j] = cluster_id
                new_neighbors = self._region_query(X, j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1

    def predict(self, X=None):
        """
        Para DBSCAN, predict simplemente devuelve los labels calculados en fit.
        """
        return np.array(self.labels_)
    def score(self, X):
        """
        Silhouette Score penalizado por cantidad de ruido.
        - Calcula Silhouette solo para puntos no-ruido.
        - Penaliza si hay muchos puntos de ruido.
        """
        X = np.asarray(X)
        labels = self.labels_

        # Separar puntos que no son ruido
        mask = labels != -1
        X_f = X[mask]
        labs = labels[mask]
        n_total = X.shape[0]
        n_f = X_f.shape[0]

        if n_f < 2:
            return 0.0

        unique_labels = np.unique(labs)
        k = len(unique_labels)
        if k < 2:
            return 0.0

        # Matriz de distancias entre todos los puntos no-ruido
        diff = X_f[:, None, :] - X_f[None, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=2))  # forma (n_f, n_f)

        silhouettes = np.zeros(n_f)
        for i in range(n_f):
            li = labs[i]

            # a(i): promedio de distancia a su mismo cluster
            same = (labs == li)
            same[i] = False  # quitarse a sí mismo
            a_i = np.mean(D[i, same]) if np.any(same) else 0.0

            # b(i): menor distancia promedio a otro cluster
            b_i = np.inf
            for lj in unique_labels:
                if lj == li:
                    continue
                other = (labs == lj)
                if np.any(other):
                    dist_prom = np.mean(D[i, other])
                    if dist_prom < b_i:
                        b_i = dist_prom

            denom = max(a_i, b_i)
            silhouettes[i] = (b_i - a_i) / denom if denom > 0 else 0.0

        silhouette_avg = np.mean(silhouettes)

        # Penalización por ruido
        penalización = n_f / n_total
        return silhouette_avg * penalización