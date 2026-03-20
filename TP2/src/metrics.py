
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


#metricas TP1

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





#metricas TP2
#ejercicio 1







def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def fscore(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    fscore = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return fscore

def auc_roc(y_true, y_prob):
    """
    Calcula el AUC de la curva ROC.
    """
    tpr, fpr = calcular_auc_ROC_PR(y_true, y_prob, curva='roc')
    auc = calcular_auc(np.sort(fpr), np.array(tpr)[np.argsort(fpr)])


    return auc


def auc_pr(y_true, y_prob):
    """
    Calcula el AUC de la curva Precision-Recall utilizando la misma estrategia que la función de graficado.
    """
    precision_vals, recall_vals = calcular_auc_ROC_PR(y_true, y_prob, curva='pr')
    auc = calcular_auc(recall_vals, precision_vals)

    return auc

def calcular_metricas(y_true, y_pred,y_prob):
    """
    Calcula y retorna las métricas de rendimiento del modelo.
    """
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = fscore(y_true, y_pred)
    roc = auc_roc(y_true, y_prob)
    pr = auc_pr(y_true, y_prob)

    resultados = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC ROC": roc,
        "AUC PR": pr
    }

    return resultados

def evaluar_modelos(y_val, resultados_dict):
    """
    resultados_dict: dict con clave = nombre del modelo, 
                     valor = (predicciones, probabilidades)
    """
    resultados = {
        nombre: calcular_metricas(y_val, preds, probs)
        for nombre, (preds, probs) in resultados_dict.items()
    }

    return pd.DataFrame(resultados).T

def calcular_auc_ROC_PR(y_true, y_prob, curva='roc'):
    """
    Calcula listas de métricas (TPR/FPR o Precision/Recall) según la curva.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    thresholds = np.concatenate(([1], thresholds, [0]))
    
    metrica_1, metrica_2 = [], []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        if curva == 'roc':
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            metrica_1.append(TPR)
            metrica_2.append(FPR)
        else:  # curva PR
            p =precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            metrica_1.append(p)
            metrica_2.append(r)
    
    return np.array(metrica_1), np.array(metrica_2)

def calcular_auc(x, y):
    """
    Calcula el área bajo la curva (AUC) utilizando la regla del trapecio.
    """
    return np.trapz(y, x)

def confusion_matrix(y_true, y_pred):
    """
    Calcula y retorna la matriz de confusión en el siguiente orden:
    [[TN, FP],
     [FN, TP]]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[TN, FP],
                   [FN, TP]])
    return cm

def plot_matrix_confusion(y_true,y_pred):
    """
    Calcula y muestra las métricas de rendimiento del modelo.
    """
    cm = confusion_matrix(y_true, y_pred)
    

    plt.figure(figsize=(6, 6))  
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()
    classes = ['Clase 0', 'Clase 1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()

def plot_curve(y_true, y_prob, curva='roc'):
    """
    Grafica curva ROC o Precision-Recall.
    """
    metrica_1, metrica_2 = calcular_auc_ROC_PR(y_true, y_prob, curva)
    
    if curva == 'roc':
        x, y = np.array(metrica_2), np.array(metrica_1)  # FPR, TPR
        xlabel, ylabel = 'False Positive Rate', 'True Positive Rate'
    else:
        x, y = np.array(metrica_2), np.array(metrica_1)  # Recall, Precision
        xlabel, ylabel = 'Recall', 'Precision'
    
    sorted_idx = np.argsort(x)
    x, y = x[sorted_idx], y[sorted_idx]


    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    if curva == 'roc':
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Curva {curva.upper()}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    
def plot_curvas(y_true, y_prob):
    """
    Calcula y grafica las curvas ROC y Precision-Recall.
    """

    plot_curve(y_true, y_prob, curva='roc')
    plot_curve(y_true, y_prob, curva='pr')

def plot_many_curves(y_test, probabilidades_dict):
    """
    Grafica múltiples curvas ROC y PR en dos subplots.
    `probabilidades_dict` debe tener como clave el nombre del método y como valor las probabilidades.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Graficar curvas ROC
    for nombre, prob in probabilidades_dict.items():
        tpr, fpr = calcular_auc_ROC_PR(y_test, prob, curva='roc')
        ax[0].plot(fpr, tpr, label=f'{nombre}')
    
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Curvas ROC')
    ax[0].legend()
    ax[0].grid()

    # Graficar curvas PR
    for nombre, prob in probabilidades_dict.items():
        precision, recall = calcular_auc_ROC_PR(y_test, prob, curva='pr')
        ax[1].plot(recall, precision, label=f'{nombre}')
    
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Curvas Precision-Recall')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()







#metricas ej 2





def precision_multiclass(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clases = np.unique(y_true)
    precisions = []
    
    for c in clases:
        # Para cada clase c, consideramos:
        # True Positives (TP): predicciones correctas de la clase c.
        # False Positives (FP): casos en que se predijo c y no es c.
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(prec)
    
        return np.mean(precisions)
        

def recall_multiclass(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clases = np.unique(y_true)
    recalls = []
    
    for c in clases:
        TP = np.sum((y_true == c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(rec)
    
    return np.mean(recalls)
    
def fscore_multiclass(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clases = np.unique(y_true)
    f_scores = []
    
    for c in clases:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f_scores.append(f)
        
    return np.mean(f_scores)

def auc_roc_multiclass(y_true, y_prob, average='macro'):
    """
    Calcula el AUC ROC en un escenario multiclase utilizando la estrategia one-vs-rest.
    
    Parámetros:
      y_true: array de etiquetas verdaderas, shape (n_samples,)
      y_prob: array de probabilidades, shape (n_samples, n_classes)
      average: 'macro' para promedio simple o 'weighted' para promedio ponderado por soporte.
      
    Retorna:
      AUC ROC promedio.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    clases = np.unique(y_true)
    aucs = []
    weights = []
    
    for i, c in enumerate(clases):
        # Convertir las etiquetas a un vector binario: 1 si es la clase c, 0 para las demás.
        binary_true = (y_true == c).astype(int)
        # Extraer la columna correspondiente a la clase c.
        prob = y_prob[:, i]
        auc_val = auc_roc(binary_true, prob)
        aucs.append(auc_val)
        weights.append(np.sum(binary_true))
    
    if average == 'macro':
        return np.mean(aucs)
    elif average == 'weighted':
        return np.average(aucs, weights=weights)
    else:
        raise ValueError("El parámetro 'average' debe ser 'macro' o 'weighted'.")

def auc_pr_multiclass(y_true, y_prob):
    """
    Calcula el AUC Precision-Recall en un escenario multiclase utilizando la estrategia one-vs-rest.
    
    Parámetros:
      y_true: array de etiquetas verdaderas, shape (n_samples,)
      y_prob: array de probabilidades, shape (n_samples, n_classes)
      average: 'macro' para promedio simple o 'weighted' para promedio ponderado por soporte.
      
    Retorna:
      AUC PR promedio.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    clases = np.unique(y_true)
    aucs = []
    
    for i, c in enumerate(clases):
        binary_true = (y_true == c).astype(int)
        prob = y_prob[:, i]
        auc_val = auc_pr(binary_true, prob)
        aucs.append(auc_val)
    
    return np.mean(aucs)



def calcular_metricas_multiclase(y_true, y_pred,y_prob):
    """
    Calcula y retorna las métricas de rendimiento del modelo.
    """

    acc = accuracy(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = fscore_multiclass(y_true, y_pred)
    roc = auc_roc_multiclass(y_true, y_prob)
    pr = auc_pr_multiclass(y_true, y_prob)

    resultados = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC ROC": roc,
        "AUC PR": pr
    }

    return resultados

def evaluar_modelos_multiclase(y_val, resultados_dict):
    """
    resultados_dict: dict con clave = nombre del modelo, 
                     valor = (predicciones, probabilidades)
    """
    resultados = {
        nombre: calcular_metricas_multiclase(y_val, preds, probs)
        for nombre, (preds, probs) in resultados_dict.items()
    }

    return pd.DataFrame(resultados).T

def confusion_matrix_multiclass(y_true, y_pred):
    """
    Calcula y retorna la matriz de confusión para clasificación multiclase.
    Cada fila representa la clase verdadera y cada columna la clase predicha.
    
    Retorna:
    - cm: matriz de confusión (np.ndarray)
    - clases: lista de clases ordenadas
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    clases = np.unique(np.concatenate((y_true, y_pred)))
    n_clases = len(clases)
    cm = np.zeros((n_clases, n_clases), dtype=int)

    clase_a_idx = {clase: idx for idx, clase in enumerate(clases)}
    
    for yt, yp in zip(y_true, y_pred):
        i = clase_a_idx[yt]
        j = clase_a_idx[yp]
        cm[i, j] += 1

    return cm, clases


def plot_confusion_matrix_multiclass(y_true, y_pred):
    """
    Calcula y muestra la matriz de confusión para clasificación multiclase.
    """
    cm, clases = confusion_matrix_multiclass(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()

    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases)
    plt.yticks(tick_marks, clases)

    # Anotar los valores en cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np


def plot_many_curves_multiclase(y_test, probabilidades_dict):
    """
    Para cada modelo en probabilidades_dict, genera:
      - Un gráfico con las curvas ROC de todas las clases.
      
    Parámetros:
      y_test: array (o iterable) de etiquetas verdaderas, shape (n_samples,)
      probabilidades_dict: diccionario con clave = nombre del modelo, 
                           valor = matriz de probabilidades (n_samples, n_clases).
                           
    Se asume que las columnas de la matriz de probabilidades están alineadas con el orden de np.unique(y_test).
    """
    y_test = np.array(y_test)
    clases = np.unique(y_test)
    count=0
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 7))

    for nombre, prob in probabilidades_dict.items():
        for col, c in enumerate(clases):
            y_bin = (y_test == c).astype(int)
            prob_c = prob[:, col]
            tpr, fpr = calcular_auc_ROC_PR(y_bin, prob_c, curva='roc')
            auc_val = calcular_auc(np.sort(fpr), np.array(tpr)[np.argsort(fpr)])
            
            ax[count].plot(fpr, tpr, label=f'Clase {c} (AUC: {auc_val:.2f})')

            ax[count].plot([0, 1], [0, 1], '--')
            ax[count].set_xlabel('FPR')
            ax[count].set_ylabel('TPR')
            ax[count].set_title(f'Curvas ROC - {nombre}')
            ax[count].legend(loc='lower right', fontsize=8)
            ax[count].grid(True)

        

        # --- Gráfico PR para el modelo 'nombre' ---
        for col, c in enumerate(clases):
            y_bin = (y_test == c).astype(int)
            prob_c = prob[:, col]
            precision_vals, recall_vals = calcular_auc_ROC_PR(y_bin, prob_c, curva='pr')
            auc_val = calcular_auc(np.sort(recall_vals), np.array(precision_vals)[np.argsort(recall_vals)])

            ax2[count].plot(recall_vals, precision_vals, label=f'Clase {c} (AUC: {auc_val:.2f})')
            ax2[count].set_xlabel('Recall')
            ax2[count].set_ylabel('Precision')
            ax2[count].set_title(f'Curvas Precision-Recall - {nombre}')
            ax2[count].legend(loc='lower left', fontsize=8)
            ax2[count].grid(True)
     

        count+=1
    plt.tight_layout()
    plt.show()
