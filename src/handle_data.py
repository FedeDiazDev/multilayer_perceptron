import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

class DatasetProcessor():
    def __init__(self, path):
        self.headers = [
            'ID', 'target', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
            'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 
            'concavity_se', 'concave_points_se', 'symmetry_se', 
            'fractal_dimension_se', 'radius_worst', 'texture_worst', 
            'perimeter_worst', 'area_worst', 'smoothness_worst', 
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 
            'symmetry_worst', 'fractal_dimension_worst'
        ]
        resolved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path)) if not os.path.isabs(path) else path
        temp_df = pd.read_csv(resolved_path, nrows=1)
        if temp_df.shape[1] == 31:
            self.df = pd.read_csv(resolved_path)
        else:
            self.df = pd.read_csv(resolved_path, names=self.headers, header=0)
        self.X_train, self.X_val = None, None
        self.y_train, self.y_val = None, None

    def pre_process(self):
        if 'ID' in self.df.columns:
            self.df = self.df.drop(columns=['ID'])
        self.df = self.df.dropna(axis=1, how='all')
        if 'target' in self.df.columns:
            if self.df['target'].dtype == object:
                self.df['target'] = self.df['target'].map({'M': 1, 'B': 0})
        self.df = self.df.dropna()

    def split(self, train_size=0.8):
        if not 0 < train_size < 1:
            raise ValueError("train_size debe estar entre 0 y 1")
        shuffled_data = self.df.sample(frac=1, random_state=42)
        split_idx = int(len(shuffled_data) * train_size)
        self.train_df = shuffled_data.iloc[:split_idx]
        self.val_df = shuffled_data.iloc[split_idx:]
        self.X_train = self.train_df.drop(columns=['target']).values
        self.y_train = self.train_df['target'].values
        self.X_val = self.val_df.drop(columns=['target']).values
        self.y_val = self.val_df['target'].values

    def visualize(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(x='target', data=self.df)
        plt.title("Distribución de Diagnósticos (0: Benigno, 1: Maligno)")
        plt.tight_layout()
        plt.savefig(resolve_path("distribution.png"))

        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title("Matriz de Correlación de Características")
        plt.tight_layout()
        plt.savefig(resolve_path("correlation.png"))

    def normalize(self):
        if self.X_train is None or self.X_val is None:
            raise RuntimeError("Primero ejecuta split() antes de normalize()")
        # Cálculo manual de la normalización Z-Score evitando fugas de datos
        self.mu = np.mean(self.X_train, axis=0)
        self.sigma = np.std(self.X_train, axis=0)
        # Prevención de división por cero en características sin varianza
        sigma_safe = np.where(self.sigma == 0, 1, self.sigma)
        # Normalización aplicada a entrenamiento y validación con los mismos parámetros
        self.X_train = (self.X_train - self.mu) / sigma_safe
        self.X_val = (self.X_val - self.mu) / sigma_safe

    def save_splits(self, train_path="../data/train.csv", test_path="../data/test.csv"):
        if not hasattr(self, 'train_df') or not hasattr(self, 'val_df'):
            raise RuntimeError("Primero ejecuta split() antes de guardar train/test")
        resolved_train = os.path.abspath(os.path.join(os.path.dirname(__file__), train_path)) if not os.path.isabs(train_path) else train_path
        resolved_test = os.path.abspath(os.path.join(os.path.dirname(__file__), test_path)) if not os.path.isabs(test_path) else test_path
        os.makedirs(os.path.dirname(resolved_train), exist_ok=True)
        os.makedirs(os.path.dirname(resolved_test), exist_ok=True)
        self.train_df.to_csv(resolved_train, index=False)
        self.val_df.to_csv(resolved_test, index=False)


if __name__ == "__main__":
    import sys
    try:
        procesador = DatasetProcessor(resolve_path("../data/data.csv"))
        print(f"Columnas originales: {procesador.df.columns[:3]}...") 
        
        # Ejecutar limpieza y renombrado
        procesador.pre_process()
        
        # YA NO USAMOS [1], USAMOS ['target'] porque ya se ha renombrado
        print("Mapeo realizado. Valores únicos:", procesador.df['target'].unique()) 
        
        procesador.visualize()
        
        # Ejecutar división
        procesador.split(train_size=0.8)
        procesador.save_splits()
        
        # AHORA SÍ: Normalizar
        procesador.normalize()
        
        # Comprobar resultados finales
        print(f"Total registros: {len(procesador.df)}")
        print(f"X_train_scaled shape: {procesador.X_train.shape}")
        print(f"y_train shape: {procesador.y_train.shape}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error al abrir el dataset: {e}", file=sys.stderr)
        sys.exit(1)