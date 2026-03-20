
import numpy as np

def divideDataFrame(df:object, ratio:float):
    """divide el dataframe en dos partes, una para entrenamiento con cierto ratio y otra para test con lo restante"""
    train = df.sample(frac=ratio, random_state=42)
    val = df.drop(train.index)
    return train, val

def one_hot(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot