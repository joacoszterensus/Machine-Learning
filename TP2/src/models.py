import numpy as np
import pandas as pd


class LogisticRegressionL2:
    def __init__(self, lr=0.001, iter=5000, lambda_=0.01):
        
        self.lr = lr
        self.iter = iter
        self.lambda_ = lambda_

        
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.X = np.c_[np.ones(X.shape[0]), X]  # bias
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.weights = np.zeros(self.n_features)

        for _ in range(self.iter):
            linear = np.dot(self.X, self.weights)
            y_pred = self.sigmoid(linear)

            # Gradiente con regularización L2 (no se regulariza el bias)
            error = y_pred - y
            gradient = (1 / self.n_samples) * np.dot(self.X.T, error)

            gradient[1:] += self.lambda_ * self.weights[1:]

            self.weights -= self.lr * gradient

    
    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class LDA:
    def __init__(self):
        self.classes_ = None    # Array con las clases únicas
        self.means_ = {}        # Diccionario: clase -> vector de media
        self.priors_ = {}       # Diccionario: clase -> probabilidad a priori
        self.cov_ = None        # Matriz de covarianza común
        self.inv_cov_ = None    # Inversa de la matriz de covarianza

    def fit(self, X, y):
        """
        Ajusta el modelo LDA.
        
        Parámetros:
          X : array-like, shape (n_samples, n_features)
          y : array-like, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calcular las medias y las probabilidades a priori para cada clase
        for c in self.classes_:
            Xc = X[y == c]
            self.means_[c] = Xc.mean(axis=0)
            self.priors_[c] = Xc.shape[0] / n_samples
        
        # Calcular la matriz de covarianza
        # Sumamos las matrices de covarianza de cada clase (sin normalizar)
        Sw = np.zeros((n_features, n_features))
        for c in self.classes_:
            Xc = X[y == c]
            centered = Xc - self.means_[c]
            Sw += centered.T @ centered
        # Dividir por (n_samples - n_classes) para obtener la covarianza común
        self.cov_ = Sw / (n_samples - len(self.classes_))
        # Calcular la inversa de la matriz de covarianza
        self.inv_cov_ = np.linalg.inv(self.cov_)

    def predict_proba(self, X):
        """
        Calcula las probabilidades de cada clase para cada muestra en X.
        
        Parámetros:
          X : array-like, shape (n_samples, n_features)
          
        Retorna:
          proba : array, shape (n_samples, n_classes)
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))
        
        # Para cada clase se calcula la función discriminante:
        # g_c(x) = xᵀ Σ⁻¹ μ₍c₎ - 0.5 * μ₍c₎ᵀ Σ⁻¹ μ₍c₎ + log(π₍c₎)
        for idx, c in enumerate(self.classes_):
            mu = self.means_[c]
            log_prior = np.log(self.priors_[c])
            # Término lineal: X @ (Σ⁻¹ μ)
            linear_term = X @ (self.inv_cov_ @ mu)
            # Término cuadrático: 0.5 * μᵀ Σ⁻¹ μ
            quad_term = 0.5 * (mu.T @ self.inv_cov_ @ mu)
            scores[:, idx] = linear_term - quad_term + log_prior
        
        # Aplicar softmax para obtener probabilidades
        # Se usa la versión estable restando el máximo por fila
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """
        Predice la clase para cada muestra en X.
        
        Parámetros:
          X : array-like, shape (n_samples, n_features)
        
        Retorna:
          preds : array, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        # Se asigna la clase con mayor probabilidad
        preds = self.classes_[np.argmax(proba, axis=1)]
        return preds


class LogisticRegressionMulticlase:
    def __init__(self, lr=0.001, iter=5000,lambda_=0.01):
        """
        Constructor de la clase.
        
        Parámetros:
        - learning_rate: Tasa de aprendizaje para el descenso de gradiente.
        - n_iterations: Número de iteraciones o epochs para el entrenamiento.
        """
        self.lr = lr
        self.iter = iter
        self.lambda_ = lambda_


    def _softmax(self, Z):
        """
        Calcula la función softmax para la matriz Z.
        
        La función softmax se define como:
            softmax(z_i) = exp(z_i) / sum(exp(z_j)) para cada fila.
        """
        # Para estabilidad numérica se resta el máximo de cada fila
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """
        Entrena el modelo usando descenso de gradiente.
        
        Parámetros:
        - X: Matriz de características de forma (n_samples, n_features).
        - y: Vector de etiquetas de forma (n_samples,).
        """
        n_samples, n_features = X.shape
        self.clases = np.unique(y)
        n_classes = len(self.clases)
        
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        
        y_onehot = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.clases):
            y_onehot[:, i] = (y == c).astype(int)

        for i in range(self.iter):
            linear = np.dot(X, self.weights) + self.bias  
            y_pred = self._softmax(linear)

            error = y_pred - y_onehot

            dW = (1 / n_samples) * np.dot(X.T, error)
            dB = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)

            self.weights -= self.lr * dW
            self.bias -= self.lr * dB
            
    
    def predict_proba(self, X):
        """
        Devuelve las probabilidades predichas para cada clase.
        
        Parámetros:
        - X: Matriz de características (n_samples, n_features).
        
        Retorna:
        - Array de probabilidades de forma (n_samples, n_classes).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)
    
    def predict(self, X):
        """
        Predice la clase a la que pertenece cada ejemplo.
        
        Parámetros:
        - X: Matriz de características (n_samples, n_features).
        
        Retorna:
        - Array de etiquetas con la clase predicha para cada ejemplo.
        """
        proba = self.predict_proba(X)

        indices_predichos = np.argmax(proba, axis=1)
        return self.clases[indices_predichos]




def entropy(y):
    """
    Calcula la entropía de un conjunto de etiquetas.
    """
    unique, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    return -np.sum(proportions * np.log2(proportions + 1e-9))

class Node:
    def __init__(self, is_leaf=False, value=None, feature=None, threshold=None, left=None, right=None):
        """
        Constructor del nodo.

        Parámetros:
        - is_leaf: Indica si el nodo es hoja.
        - value: Valor asignado en caso de nodo hoja (etiqueta más común).
        - feature: Índice de la característica usada para la división.
        - threshold: Umbral de la característica.
        - left: Nodo hijo izquierdo.
        - right: Nodo hijo derecho.
        """
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, num_thresholds=10):
        """
        Constructor del árbol de decisión.

        Parámetros:
        - max_depth: Profundidad máxima del árbol.
        - min_samples_split: Número mínimo de muestras requerido para dividir un nodo.
        - num_thresholds: Número de umbrales candidatos a evaluar por cada característica.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_thresholds = num_thresholds
        self.root = None

    def fit(self, X, y):
        """
        Ajusta el árbol de decisión a los datos X y etiquetas y.
        """
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        # Condición de parada: profundidad máxima, muestras insuficientes o etiquetas homogéneas.
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = self._most_common_label(y)
            return Node(is_leaf=True, value=leaf_value)
        
        best_feature = None
        best_thresh = None
        best_gain = -np.inf
        current_entropy = entropy(y)
        
        for feature in range(num_features):
            X_column = X.iloc[:, feature]
            min_val, max_val = X_column.min(), X_column.max()
            if min_val == max_val:
                continue  # No se puede dividir si la característica es constante.
            thresholds = np.linspace(min_val, max_val, self.num_thresholds)
            
            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = X_column > thresh
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = current_entropy - (left_mask.sum()/num_samples)*entropy(y_left) \
                       - (right_mask.sum()/num_samples)*entropy(y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = thresh
        
        # Si no se encuentra una buena división, se crea un nodo hoja.
        if best_gain == -np.inf:
            leaf_value = self._most_common_label(y)
            return Node(is_leaf=True, value=leaf_value)
        
        # Construir recursivamente los subárboles.
        X_column = X.iloc[:, best_feature]
        left_mask = X_column <= best_thresh
        right_mask = X_column > best_thresh

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(is_leaf=False, feature=best_feature, threshold=best_thresh,
                    left=left_subtree, right=right_subtree)
    
    def _most_common_label(self, y):
        """
        Retorna la etiqueta más frecuente en y.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def _traverse_tree(self, x, node):
        """
        Recorre el árbol para predecir la etiqueta de una muestra x.
        Se espera que x sea un vector de NumPy o una Series.
        """
        if node.is_leaf:
            return node.value
        if isinstance(x, pd.Series):
            x_val = x.to_numpy()
        else:
            x_val = x
        
        if x_val[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """
        Predice las etiquetas para el conjunto de datos X.
        Se espera que X sea un DataFrame.
        """
        X_values = X.to_numpy()
        return np.array([self._traverse_tree(x, self.root) for x in X_values])
    
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, sample_size=None, num_thresholds=10):
        """
        Constructor del Bosque Aleatorio.

        Parámetros:
        - n_estimators: Número de árboles en el bosque.
        - max_depth: Profundidad máxima para cada árbol.
        - min_samples_split: Número mínimo de muestras para dividir un nodo.
        - sample_size: Tamaño de la muestra bootstrap para cada árbol. Si es None, se usa todo el dataset.
        - num_thresholds: Número de umbrales candidatos a evaluar en cada nodo.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.num_thresholds = num_thresholds
        self.trees = []
        self.classes = None  # Almacena las clases presentes en el dataset

    def fit(self, X, y):
        """
        Entrena el Bosque Aleatorio.
        Se espera que X sea un DataFrame y y una Series.
        """
        self.trees = []
        n_samples = X.shape[0]
        if self.sample_size is None:
            self.sample_size = n_samples

        self.classes = np.unique(y)
            
        for _ in range(self.n_estimators):
            # Seleccionar muestras bootstrap usando .iloc con índices.
            indices = np.random.choice(n_samples, self.sample_size, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split,
                                num_thresholds=self.num_thresholds)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict_proba(self, X):
        """
        Retorna la probabilidad de cada clase para cada muestra en X.
        Se espera que X sea un DataFrame.

        Para cada muestra, se calcula el porcentaje de votos que recibió cada clase
        de entre los árboles del bosque.
        """
        # Obtener las predicciones de todos los árboles
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes)))
        
        for i in range(n_samples):
            # Para cada muestra, contar los votos de cada clase
            for j, cls in enumerate(self.classes):
                proba[i, j] = np.sum(tree_preds[:, i] == cls) / self.n_estimators
        return proba
    
    def predict(self, X):
        """
        Predice la etiqueta para cada muestra en X.
        Para cada muestra se devuelve la clase que obtuvo la mayor probabilidad,
        de acuerdo al cálculo realizado en predict_proba.
        Se espera que X sea un DataFrame.
        """
        proba = self.predict_proba(X)
        # Seleccionar la clase con mayor probabilidad para cada muestra
        predictions = [self.classes[np.argmax(proba[i])] for i in range(proba.shape[0])]
        return np.array(predictions)
