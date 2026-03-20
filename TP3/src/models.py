import numpy as np
import matplotlib.pyplot as plt

from .ActivationFunction import relu, relu_derivative, softmax, softmax_derivative
from .data_Splitting import one_hot
from .metrics import cross_entropy_loss,accuracy, confusion_matrix_multiclass

class NeuralNetwork:
    
    def __init__(self, layer_sizes, lr=0.01, 
                 lr_schedule=None, 
                 schedule_params=None, 
                 size_batch=None,
                 optimizer='sgd',
                 beta1=0.9,
                 beta2=0.999,         
                 L2_lambda=None,
                 early_stopping=False,
                 patience=10,
                 dropout_prob=None,
                 Print=True,
                 seed=42
                 ):
        np.random.seed(seed)

        self.Print = Print

        # Learning rate and scheduler
        self.initial_lr = lr
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.schedule_params = schedule_params or {}

        # Batch size
        self.size_batch = size_batch if size_batch is not None and size_batch > 0 else None

        # Optimizer settings
        self.optimizer = optimizer.lower()
        if self.optimizer not in ['sgd', 'adam']:
            raise ValueError(f"Optimizer '{optimizer}' no soportado. Elige 'sgd' o 'adam'.")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8
        self.t = 0

        # L2 regularization
        self.l2_lambda = L2_lambda

        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience

        # Dropout
        self.dropout_prob = dropout_prob
        self.training = False

        # Initialize weights and biases
        self.n_weight_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.n_weight_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            std = np.sqrt(2.0 / n_in)
            self.weights.append(np.random.normal(0, std, (n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

        # Adam moments
        if self.optimizer == 'adam':
            self.m_w = [np.zeros_like(W) for W in self.weights]
            self.v_w = [np.zeros_like(W) for W in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]


    def forward(self, X):
        A = X
        activations = [X]
        pre_activations = []
        self.dropout_masks = []

        # Hidden layers with optional dropout
        for i in range(self.n_weight_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = relu(Z)

            if self.training and self.dropout_prob:
                mask = (np.random.rand(*A.shape) > self.dropout_prob) / (1 - self.dropout_prob)
                A *= mask
                self.dropout_masks.append(mask)

            else:
                self.dropout_masks.append(None)

            pre_activations.append(Z)
            activations.append(A)

        # Output layer (no dropout)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)

        pre_activations.append(Z)
        activations.append(A)

        return activations, pre_activations



    def backward(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        y_onehot = one_hot(y_true, activations[-1].shape[1])
        grads_w = [None] * self.n_weight_layers
        grads_b = [None] * self.n_weight_layers
        
        # Gradiente inicial en la capa de salida
        dZ = softmax_derivative(pre_activations[-1], y_onehot)
        
        # Recorro las capas hacia atrás
        for i in reversed(range(self.n_weight_layers)):
            A_prev = activations[i]  # activación de la capa anterior

            # 1) Gradientes de pesos y biases
            grads_w[i] = A_prev.T @ dZ / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # 1.1) Regularización L2 si aplica
            if self.l2_lambda:
                grads_w[i] += (self.l2_lambda / m) * self.weights[i]

            # 2) Preparo dZ para la siguiente iteración (si no estoy en la capa 0)
            if i > 0:
                # 2.1) Gradiente sobre la activación previa
                dA_prev = dZ @ self.weights[i].T

                # 2.2) Aplico la máscara de dropout (inverted dropout)
                mask = self.dropout_masks[i-1]
                if mask is not None:
                    dA_prev *= mask

                # 2.3) Paso por la derivada de ReLU
                dZ = dA_prev * relu_derivative(pre_activations[i-1])

        # 3) Actualización de parámetros (SGD o Adam)
        self.t += 1
        for i in range(self.n_weight_layers):
            if self.optimizer == 'sgd':
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i]  -= self.lr * grads_b[i]
            else:
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i]**2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i]**2)
                m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)
                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
                self.biases[i]  -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)                
                pass


    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        m = X_train.shape[0]
        self.history = {'train_loss': [], 'val_loss': [],'val_acc':[], 'lrs': []}

        # early stopping
        if self.early_stopping :
            best_val_loss, no_improve = float('inf'), 0
            best_w = None; best_b = None
        self.training = True

        for epoch in range(epochs):
            # shuffle
            perm = np.random.permutation(m)
            X_sh, y_sh = X_train[perm], y_train[perm]
            epoch_loss = 0

            # iterate batches
            if self.size_batch:
                for start in range(0, m, self.size_batch):
                    
                    end = start + self.size_batch
                    Xb, yb = X_sh[start:end], y_sh[start:end]
                    
                    acts, pre = self.forward(Xb)
                    batch_loss = cross_entropy_loss(yb, acts[-1], self.weights, self.l2_lambda)
                    epoch_loss += batch_loss * Xb.shape[0]
                    self.backward(acts, pre, yb)

                epoch_loss /= m
            else:
                acts, pre = self.forward(X_train)
                epoch_loss = cross_entropy_loss(y_train, acts[-1], self.weights, self.l2_lambda)
                self.backward(acts, pre, y_train)

            self.history['train_loss'].append(epoch_loss)

            # lr schedule
            if self.lr_schedule:
                self.lr = self.lr_schedule(epoch, **self.schedule_params)
            self.history['lrs'].append(self.lr)
            
            # validation
            val_probs = self.forward(X_val)[0][-1]
            val_loss = cross_entropy_loss(y_val, val_probs, self.weights, self.l2_lambda)
            self.history['val_loss'].append(val_loss)

            val_acc = accuracy(y_val, np.argmax(val_probs, axis=1))
            self.history['val_acc'].append(val_acc)


            # early stopping
            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                    best_w = [W.copy() for W in self.weights]
                    best_b = [b.copy() for b in self.biases]
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        for _ in range(self.patience):
                            self.history['val_loss'].pop()
                            self.history['val_acc'].pop()
                            self.history['train_loss'].pop()
                            self.history['lrs'].pop()
                        self.weights, self.biases = best_w, best_b
                        break
            # print
            msg = f"Epoch {epoch}: Train Loss={epoch_loss:.4f}"
            if epoch % 10 == 0 and self.Print:
                acc_val = accuracy(y_val, np.argmax(val_probs, axis=1))
                msg += f", Val Loss={val_loss:.4f}, Val Acc={acc_val:.4f}, LR={self.lr:.6f}"
                print(msg)
        self.training = False


    def predict(self, X):
        self.training = False
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        self.training = False
        activations, _ = self.forward(X)
        return activations[-1]

    def grafico_loss(self):
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.show()

    def metrics(self):

        print("Train Loss:", self.history['train_loss'][-1])
        print("Validation Loss:", self.history['val_loss'][-1])
        print("Validation Accuracy:", self.history['val_acc'][-1])


        
