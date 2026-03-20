import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=None):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout_prob:
                    layers.append(nn.Dropout(p=dropout_prob))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PytorchModel:
    """
    Clase para crear y entrenar un modelo de red neuronal en PyTorch con soporte para
    early stopping, regularización, optimizadores y programación de learning rate.
    """
    def __init__(self,
                 layer_sizes,
                 lr=1e-3,
                 size_batch=None,
                 optimizer='sgd',
                 beta1=0.9,
                 beta2=0.999,
                 L2_lambda=None,
                 early_stopping=False,
                 patience=10,
                 dropout_prob=None,
                 lr_schedule=None,
                 schedule_params=None,
                 seed=42
                 ):
        

        
        torch.manual_seed(seed)
        self.model = NeuralNet(layer_sizes, dropout_prob)

        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = size_batch or 64
        self.initial_lr = lr

        # Optimizador
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=L2_lambda or 0.0)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=beta1, weight_decay=L2_lambda or 0.0)

        self.early_stopping = early_stopping
        self.patience = patience

        # LR Scheduler
        self.lr_schedule = lr_schedule
        self.schedule_params = schedule_params or {}

    def fit(self, x_train, y_train, x_val, y_val, max_epochs=100):
        # Conversión a tensores
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).long()
        x_val   = torch.from_numpy(x_val).float()
        y_val   = torch.from_numpy(y_val).long()

        # DataLoader
        train_ds = TensorDataset(x_train, y_train)
        val_ds   = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': [],'val_acc':[], 'lrs': []}

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = 0.0

            # Actualizar el learning rate si hay scheduler definido
            if self.lr_schedule:
                lr = self.lr_schedule(epoch - 1, **self.schedule_params)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                history['lrs'].append(lr)
            else:
                history['lrs'].append(self.initial_lr)

            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.criterion(output, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validación
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct  = 0
                total    = 0

                for xb, yb in val_loader:
                    output = self.model(xb)
                    loss   = self.criterion(output, yb)
                    val_loss += loss.item()

                    # Cálculo de accuracy por batch
                    preds    = output.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += yb.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc      = correct / total

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)


            if epoch % 10 == 0:
                print(f"EPOCH:{epoch}, Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            # Early stopping
            if self.early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    
        self.historia = history

    def predict(self, x):
        self.model.eval()
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits, dim=1).numpy()

    def predict_proba(self, x):
        """Devuelve el vector de probabilidades softmax para cada muestra."""
        self.model.eval()
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.model(x)
            probs  = F.softmax(logits, dim=1)
            return probs.numpy()
        
    def weigths(self):
        """Devuelve los pesos de la red neuronal."""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.numpy())
        return weights
        
    def grafico_loss(self):
        plt.plot(self.historia['train_loss'], label='Train Loss')
        if 'val_loss' in self.historia:
            plt.plot(self.historia['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.show()
    
    def metrics(self):
        print("Train Loss:", self.historia['train_loss'][-1])
        print("Validation Loss:", self.historia['val_loss'][-1])
        print("Validation Accuracy:", self.historia['val_acc'][-1])

    