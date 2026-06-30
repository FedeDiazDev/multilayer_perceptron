import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from handle_data import DatasetProcessor
from multilayer_perceptron import MultilayerPerceptron

np.random.seed(42)

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def train_model():
    try:
        procesador = DatasetProcessor(resolve_path("../data/data.csv"))
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error al abrir el dataset: {e}", file=sys.stderr)
        sys.exit(1)

    procesador.pre_process()
    procesador.split(train_size=0.8)
    procesador.save_splits() 
    procesador.normalize() 
    np.save(resolve_path("scaling_params.npy"), {'mu': procesador.mu, 'sigma': procesador.sigma})

    dims = [30, 16, 16, 2]
    epochs = 2000
    lr = 0.05
    patience = 100
    
    mlp = MultilayerPerceptron(dims)
    Y_train_oh = mlp.to_one_hot(procesador.y_train)
    Y_val_oh = mlp.to_one_hot(procesador.y_val)

    print(f"Iniciando entrenamiento: {epochs} épocas máximas...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_params = {k: v.copy() for k, v in mlp.params.items()}

    for i in range(epochs):
        Y_hat, cache = mlp.forward_propagation(procesador.X_train)
        loss = mlp.compute_loss(Y_hat, Y_train_oh)
        
        Y_hat_val, _ = mlp.forward_propagation(procesador.X_val)
        val_loss = mlp.compute_loss(Y_hat_val, Y_val_oh)
        
        train_acc = mlp.get_accuracy(Y_hat, procesador.y_train)
        val_acc = mlp.get_accuracy(Y_hat_val, procesador.y_val)
        
        mlp.history['loss'].append(loss)
        mlp.history['val_loss'].append(val_loss)
        mlp.history['acc'].append(train_acc)
        mlp.history['val_acc'].append(val_acc)
        
        # Lógica de Early Stopping para evitar el Overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Guardamos una copia exacta de los pesos en la mejor época
            best_params = {k: v.copy() for k, v in mlp.params.items()}  # Estado exacto evaluado
        else:
            patience_counter += 1

        print(
            f"Época {i + 1:04d}/{epochs} - "
            f"loss: {loss:.4f} - val_loss: {val_loss:.4f} - "
            f"acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
        )

        if patience_counter >= patience:
            print(f"Early Stopping activado en la época {i}. Restaurando mejores pesos.")
            break

        grads = mlp.backward_propagation(Y_train_oh, cache)
        mlp.update_parameters(grads, lr)

    # Restaurar siempre el mejor estado encontrado en validación
    mlp.params = best_params

    # Guardar modelo con configuración
    config = {'architecture': dims, 'epochs_run': len(mlp.history['loss']), 'learning_rate': lr, 'best_val_loss': best_val_loss}
    mlp.save_model(resolve_path("modelo_tfm.npy"), config=config)
    print("Entrenamiento finalizado y modelo guardado.")
    
    plot_results(mlp.history)

def plot_results(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Entrenamiento')
    plt.plot(history['val_loss'], label='Validación')
    plt.title('Evolución de la Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Error')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Entrenamiento')
    plt.plot(history['val_acc'], label='Validación')
    plt.title('Evolución de la Precisión (Accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(resolve_path("resultados_entrenamiento.png"))
    plt.show()

if __name__ == "__main__":
    train_model()