import numpy as np
import pandas as pd



class LinearRegression:
    def __init__(self, x, y):
        self.featuresNames=x.iloc[0].index.tolist()
        
        self.x = x.to_numpy() 
        self.y = y
    
        self.coefPseudoInversa = None  
        self.coefGradientDescent=None

    def pseudoInversaTraining(self,L2=0):
        """Calcula los coeficientes del modelo usando la pseudo-inversa para un polinomio de grado M."""
        
        xPoly=np.hstack([np.ones((self.x.shape[0], 1)), self.x])

        self.coefPseudoInversa = getBestCoef(xPoly, self.y, L2)
    
    def visualizePseudoInversaCoef(self):
        """Imprime los coeficientes del modelo entrenado con la pseudoinversa."""
        if self.coefPseudoInversa is None:
            print("El modelo aún no ha sido entrenado con la pseudoinversa.")
            return

        coefDf = dataFrameCoef(self.coefPseudoInversa, self.featuresNames)

        print("\nCoeficientes del modelo (Pseudoinversa):")
        print(coefDf.to_string(index=False))
        return 
    
    def predictPseudoInversa(self, x):
        """Predice la salida para un conjunto de datos x usando la pseudoinversa."""
        x = x.to_numpy()  

        xPoly=np.hstack([np.ones((x.shape[0], 1)), x])

        y = xPoly @ self.coefPseudoInversa
        return y

    def getCoefPseudoInversa(self):
        """Devuelve los coeficientes de la pseudoinversa del modelo."""
        return self.coefPseudoInversa

    def gradientDescentTraining(self, data=None, target=None, step=0.01, iter=5000,L2=0, L1=0):   
        """Entrena el modelo usando gradiente descendente con regularizacion o sin regularizacion."""

        if data is None:
            data = self.x
        if target is None:
            target = self.y

        x_poly = np.hstack([np.ones((data.shape[0], 1)), data])

        w = gradienteDescendiente(x_poly, target, np.zeros(x_poly.shape[1]), 
                                        step, iter, L2, L1)
        self.coefGradientDescent = w

    def visualizeGradientDescentCoef(self):
        """Imprime los coeficientes del modelo entrenado con gradiente descendente."""
        if self.coefGradientDescent is None:
            print("El modelo aún no ha sido entrenado con la gradiente descendiente.")
            return

        coefDf = dataFrameCoef(self.coefGradientDescent, self.featuresNames)

        print("\nCoeficientes del modelo (gradiente descendiente):")
        print(coefDf.to_string(index=False))
    
    def getCoefGradientDescent(self):
        """Devuelve los coeficientes de gradiente descendente del modelo."""
        return self.coefGradientDescent
    
    def predictGradientDescent(self, x):
        """Predice la salida para un conjunto de datos x usando gradiente descendente."""
        x = x.to_numpy()  
        xPoly=np.hstack([np.ones((x.shape[0], 1)), x])

        y = xPoly @ self.coefGradientDescent
        return y    

def getBestCoef(X, Y, L2=0):

    """Calcula los coeficientes óptimos usando la pseudo-inversa con regularización L2 (Ridge Regression)."""
    n_features = X.shape[1]  # Cantidad de características
    I = np.eye(n_features)  # Matriz identidad del tamaño de XTX
    if L2 > 0:
        XTX=X.T @ X
        XTY = X.T @ Y
        w_ridge = np.linalg.inv(XTX + L2 * I) @ XTY
    else:
        w_ridge = np.linalg.pinv(X) @ Y  

    return w_ridge

def gradienteDescendiente(A, b, x_inicial, step, iter, L2=0, L1=0):
    x = x_inicial
    m=A.shape[0]
    for i in range(iter):
        gradient = (2/m) * A.T @ (A @ x - b)  

        if not(L2 or L1):
            x = x - step * gradient

        elif L2:  
            gradient += 2 * L2 * x  
            x = x - step * gradient

        elif L1:  
            gradient += L1 * np.sign(x) 
            x = x - step * gradient

    return x

def dataFrameCoef(coef, feature_names):
    """Crea un DataFrame con los coeficientes del modelo."""

    num_features = len(feature_names)
    M = (len(coef) - 1) // num_features  

    coef_dict = { "Feature": ["Bias"] + feature_names }  

    coef_dict["Grado 0"] = [coef[0]] + ["-"] * num_features  
    
    for degree in range(1, M + 1):
        coef_dict[f"Grado {degree}"] = ["-"]  
        for i in range(num_features):
            index = 1 + i + (degree - 1) * num_features  
            coef_dict[f"Grado {degree}"].append(coef[index])
    coef_df = pd.DataFrame(coef_dict)
    return coef_df
