import numpy as np

class MultilayerPerceptron:
    def __init__(self, layer_size):
        self.weights = []
        self.biases = []
        self.params = {}
        self.L = len(layer_size) - 1
        self.history = {'loss' : [], 'val_loss' : [], 'acc': [], 'val_acc' : []}
        for i in range(1, self.L + 1):
            self.params[f'W{i}'] = np.random.randn(layer_size[i-1], layer_size[i]) * np.sqrt(2 / layer_size[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_size[i]))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        cache = {'A0': X}
        for i in range(1, self.L):
            # Suma ponderada y activación Sigmoide para capas ocultas
            Z = np.dot(cache[f'A{i-1}'], self.params[f'W{i}']) + self.params[f'b{i}']
            cache[f'A{i}'] = self.sigmoid(Z)
        # Capa de salida con activación Softmax para distribución de probabilidad
        Z_last = np.dot(cache[f'A{self.L-1}'], self.params[f'W{self.L}']) + self.params[f'b{self.L}']
        cache[f'A{self.L}'] = self.softmax(Z_last)
        return cache[f'A{self.L}'], cache
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def backward_propagation(self, Y, cache):
        m = Y.shape[0]
        grads = {}
        # 1. Derivada de la pérdida respecto a la salida (Softmax + Entropía Cruzada simplificada)
        dZ = cache[f'A{self.L}'] - Y
        # 2. Retropropagación del error aplicando la regla de la cadena (desde la capa L hasta la 1)
        for i in range(self.L, 0, -1):
            # Gradientes respecto a los pesos (dW) y sesgos (db) de la capa actual
            grads[f'dW{i}'] = (1/m) * np.dot(cache[f'A{i-1}'].T, dZ)
            grads[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            # Propagación del error hacia la capa oculta anterior (si no estamos en la entrada)
            if i > 1:
                # Derivada de la sigmoide: A * (1 - A)
                dA_prev = np.dot(dZ, self.params[f'W{i}'].T)
                A_prev = cache[f'A{i-1}']
                
                # Multiplicación por la derivada de la función de activación Sigmoide: A * (1 - A)
                dZ = dA_prev * (A_prev * (1 - A_prev))
        return grads
        
    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        # Añadimos un valor minúsculo (1e-15) para evitar el log(0) que daría error
        loss = - (1/m) * np.sum(Y * np.log(Y_hat + 1e-15))
        return loss
    
    def to_one_hot(self, y, num_classes=2):
        return np.eye(num_classes)[y.astype(int)]
    
    def update_parameters(self, grads, learning_rate):
        for i in range(1, self.L + 1):
            self.params[f'W{i}'] -= learning_rate * grads[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * grads[f'db{i}']
            
    def predict(self, X):
        # Retorna la clase con mayor probabilidad (0 o 1)
        Y_hat, _ = self.forward_propagation(X)
        return np.argmax(Y_hat, axis=1)

    def get_accuracy(self, Y_hat, Y_real_labels):
        predictions = np.argmax(Y_hat, axis=1)
        return np.mean(predictions == Y_real_labels)

    def save_model(self, filename="model.npy", config=None):
        payload = {"params": self.params, "config": config}
        np.save(filename, payload, allow_pickle=True)

    def load_model(self, filename="model.npy"):
        data = np.load(filename, allow_pickle=True).item()
        # Compatibilidad: permite cargar modelos antiguos que guardaban solo params
        if isinstance(data, dict) and "params" in data:
            self.params = data["params"]
            return data.get("config")
        self.params = data
        return None