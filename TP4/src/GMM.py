import numpy as np
from src.Kmeans import KMeans
def _log_multivariate_normal_density(X, means, covariances):
    # X: (N, D), means: (K, D), covariances: (K, D, D)
    N, D = X.shape
    K = means.shape[0]

    # Precomputar inversas y determinantes logarítmicos
    inv_covs = np.linalg.inv(covariances)      # (K, D, D)
    log_det_covs = np.log(np.linalg.det(covariances) + 1e-10)  # (K,)

    diff = X[:, None, :] - means[None, :, :]   # (N, K, D)
    m_dist = np.einsum('nkd,kde,nke->nk', diff, inv_covs, diff)  # Mahalanobis

    # log densidad
    log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det_covs)[None, :]  # (1, K)
    log_pdf = log_norm - 0.5 * m_dist  # (N, K)

    return np.exp(log_pdf)  # Convertimos de nuevo para mantener compatibilidad


class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4,
                 random_state=0, reg_covar=1e-6, verbose=False):
        self.n_components   = n_components
        self.max_iter       = max_iter
        self.tol            = tol
        self.random_state   = random_state
        self.reg_covar      = reg_covar
        self.verbose        = verbose

        # Parámetros
        self.means_         = None  # (K, D)
        self.covariances_   = None  # (K, D, D)
        self.weights_       = None  # (K,)
        self.resp_          = None  # (N, K)
        self.log_likelihood_= None
        self.labels_        = None  # (N,)

    def _initialize_parameters(self, X):
        N, D = X.shape
        # Inicializar con KMeans optimizado
        km = KMeans(
            n_clusters=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        km.fit(X)
        self.means_ = km.centroids_.copy()
        labels = km.labels_

        # Pesos
        self.weights_ = np.bincount(labels, minlength=self.n_components) / N

        # Covarianzas iniciales
        cov_global = np.cov(X, rowvar=False)
        self.covariances_ = np.array([
            np.cov(X[labels == k], rowvar=False) + self.reg_covar * np.eye(D)
            if np.sum(labels == k) > 1 else cov_global + self.reg_covar * np.eye(D)
            for k in range(self.n_components)
        ])

    def _e_step(self, X):
        # Densidades multivariadas: (N, K)
        pdf = _log_multivariate_normal_density(X, self.means_, self.covariances_)
        weighted = pdf * self.weights_[None, :]
        total = weighted.sum(axis=1, keepdims=True)
        resp = weighted / (total + 1e-300)
        return resp, total.squeeze()

    def _m_step(self, X, resp):
        N, D = X.shape
        K = self.n_components
        Nk = resp.sum(axis=0)  # (K,)

        # Corregir componentes muertas
        eps = 1e-6
        cov_global = np.cov(X, rowvar=False)
        for k in range(K):
            if Nk[k] < eps:
                Nk[k] = eps
                if self.verbose:
                    print(f"Componente {k} muerta, reinicializando...")
                idx = np.random.randint(N)
                self.means_[k] = X[idx]
                self.covariances_[k] = cov_global + self.reg_covar * np.eye(D)

        # Pesos
        self.weights_ = Nk / Nk.sum()
        # Medias
        self.means_ = (resp.T @ X) / Nk[:, None]
        # Covarianzas
        diff = X[:, None, :] - self.means_[None, :, :]  # (N, K, D)
        covs = np.einsum('nk,nkd,nke->kde', resp, diff, diff) / Nk[:, None, None]
        covs += self.reg_covar * np.eye(D)[None, :, :]
        self.covariances_ = covs

    def fit(self, X):
        X = np.asarray(X)
        self._initialize_parameters(X)
        prev_ll = None

        for it in range(self.max_iter):
            resp, totals = self._e_step(X)
            self._m_step(X, resp)
            # log-likelihood usando log total
            ll = np.sum(np.log(totals + 1e-300))
            if self.verbose:
                print(f"Iter {it:03d} - logL = {ll:.6f}")
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.resp_ = resp
        self.log_likelihood_ = ll
        self.labels_ = resp.argmax(axis=1)
        return self

    def predict(self, X):
        resp, _ = self._e_step(np.asarray(X))
        return resp.argmax(axis=1)
    def score(self, X):
        """
        Devuelve la log-verosimilitud promedio de las muestras X
        """
        X = np.asarray(X)
        _, totals = self._e_step(X)
        # totals es la suma de densidades ponderadas;  log(totals) da  log-prob por muestra
        ll_total = np.sum(np.log(totals + 1e-300))
        return ll_total / X.shape[0]