import argparse
import os

import numpy as np

from handle_data import DatasetProcessor
from multilayer_perceptron import MultilayerPerceptron


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def load_scaling_params(path):
    data = np.load(path, allow_pickle=True).item()
    return data["mu"], data["sigma"]


def load_dataset(path, mu, sigma):
    procesador = DatasetProcessor(path)
    procesador.pre_process()

    has_target = "target" in procesador.df.columns
    if has_target:
        features = procesador.df.drop(columns=["target"]).values
        labels = procesador.df["target"].values.astype(int)
    else:
        features = procesador.df.values
        labels = None

    sigma_safe = np.where(sigma == 0, 1, sigma)
    features = (features - mu) / sigma_safe
    return features, labels


def infer_architecture(payload):
    config = payload.get("config") if isinstance(payload, dict) else None
    if config and "architecture" in config:
        return config["architecture"]

    params = payload["params"] if isinstance(payload, dict) and "params" in payload else payload
    layer_count = len([key for key in params.keys() if key.startswith("W")])
    if layer_count == 0:
        raise ValueError("No se pudo inferir la arquitectura del modelo.")

    architecture = []
    for index in range(1, layer_count + 1):
        weights = params[f"W{index}"]
        if index == 1:
            architecture.append(weights.shape[0])
        architecture.append(weights.shape[1])
    return architecture


def binary_cross_entropy(y_true, y_prob):
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

# Cálculo manual de la matriz de confusión sobre predicciones binarias
def confusion_matrix_binary(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0)) # Verdaderos Negativos
    fp = np.sum((y_true == 0) & (y_pred == 1)) # Falsos Positivos
    fn = np.sum((y_true == 1) & (y_pred == 0)) # Falsos Negativos
    tp = np.sum((y_true == 1) & (y_pred == 1)) # Verdaderos Positivos
    
    return np.array([[tn, fp], [fn, tp]])


def classification_metrics(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def roc_auc_score_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)

    return float(np.trapezoid(tpr, fpr))


def run_prediction(data_path, model_path, scaling_path):
    data_path = resolve_path(data_path)
    model_path = resolve_path(model_path)
    scaling_path = resolve_path(scaling_path)

    mu, sigma = load_scaling_params(scaling_path)

    payload = np.load(model_path, allow_pickle=True).item()
    architecture = infer_architecture(payload)

    mlp = MultilayerPerceptron(architecture)
    mlp.load_model(model_path)

    X, y_true = load_dataset(data_path, mu, sigma)
    y_pred_proba = mlp.forward_propagation(X)[0][:, 1]
    y_pred = mlp.predict(X)

    print(f"x_shape : {X.shape}")
    
    if y_true is not None:
        accuracy = np.mean(y_pred == y_true)
        loss = binary_cross_entropy(y_true, y_pred_proba)
        metrics = classification_metrics(y_true, y_pred)
        roc_auc = roc_auc_score_binary(y_true, y_pred_proba)

        print(f"accuracy: {accuracy:.4f}")
        print(f"binary_cross_entropy: {loss:.4f}")
        print("confusion_matrix:")
        print(metrics["confusion_matrix"])
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall: {metrics['recall']:.4f}")
        print(f"specificity: {metrics['specificity']:.4f}")
        print(f"f1_score: {metrics['f1']:.4f}")
        print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"roc_auc: {roc_auc:.4f}")
        return y_pred, y_pred_proba, accuracy, loss
    else:
        print("No target column found. Predictions only:")
        for i, pred in enumerate(y_pred[:10]):  # Mostrar primeras 10
            print(f"  Sample {i}: {pred}")
        return y_pred, y_pred_proba, None, None


def build_parser():
    parser = argparse.ArgumentParser(description="Predictor para el multilayer perceptron")
    parser.add_argument("--data", default="../data/data.csv", help="Ruta al CSV a evaluar")
    parser.add_argument("--model", default="modelo_tfm.npy", help="Ruta al modelo guardado")
    parser.add_argument("--scaling", default="scaling_params.npy", help="Ruta a los parámetros de escalado")
    return parser


def main():
    args = build_parser().parse_args()
    run_prediction(args.data, args.model, args.scaling)


if __name__ == "__main__":
    main()