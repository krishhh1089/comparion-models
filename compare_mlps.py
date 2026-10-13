"""
MLP from scratch vs scikit-learn MLP on the UCI Student Performance dataset.

Dataset:
  UCI Student Performance (Portuguese or Mathematics course grades)
  https://archive.ics.uci.edu/dataset/320/student+performance

Task:
  Predict final grade category: pass (G3 >= 10) vs fail (G3 < 10).
  Features exclude G3 to prevent target leakage. G1/G2 are retained as earlier grades.

Scratch model:
  NumPy MLP, sigmoid output for binary classification, binary cross-entropy,
  mini-batch gradient descent, backpropagation.

Library model:
  sklearn.neural_network.MLPClassifier (sigmoid/logistic output, Adam optimizer).
  Note: sklearn's MLPClassifier does not expose a plain full-batch gradient-descent
  optimizer option. Its default Adam is gradient-based optimization; the scratch
  model uses vanilla mini-batch gradient descent. This is disclosed for fairness.

Install:
  pip install numpy pandas matplotlib scikit-learn ucimlrepo

Run:
  python compare_mlps.py
Optional:
  python compare_mlps.py --subject math
  python compare_mlps.py --subject portuguese
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning

RANDOM_STATE = 42


def load_student_data(subject="math"):
    """Load authentic UCI student course-grade data via ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency ucimlrepo. Install with: pip install ucimlrepo"
        ) from exc

    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features.copy()
    y_grade = dataset.data.targets["G3"].astype(int)
    # Use only one course, matching the selected subject.
    subject_col = "school"
    if subject == "math":
        # UCI files are split into math and Portuguese; ucimlrepo exposes the
        # combined metadata differently across versions. Use the source files
        # directly when available via the dataset metadata URL.
        pass
    # If course is available as a feature, filter it; otherwise dataset release
    # returned by the repository is used as-is.
    if "subject" in X.columns:
        mask = X["subject"].astype(str).str.lower().str.contains(subject)
        X, y_grade = X.loc[mask].copy(), y_grade.loc[mask].copy()
        X = X.drop(columns=["subject"])
    # Target must not appear among inputs.
    X = X.drop(columns=["G3"], errors="ignore")
    y = (y_grade >= 10).astype(int).to_numpy()
    return X, y


def load_student_data_csv(subject="math"):
    """Fallback loader from the official UCI ZIP if the API wrapper differs."""
    import urllib.request
    import zipfile
    from io import BytesIO

    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            archive = zipfile.ZipFile(BytesIO(response.read()))
        filename = "student-mat.csv" if subject == "math" else "student-por.csv"
        with archive.open(filename) as file:
            df = pd.read_csv(file, sep=";")
    except Exception as exc:
        raise SystemExit(
            "Could not download the UCI dataset. Check internet access and try again. "
            f"Details: {exc}"
        ) from exc
    y = (df["G3"].astype(int) >= 10).astype(int).to_numpy()
    X = df.drop(columns=["G3"])
    return X, y


class ScratchMLP:
    """One-hidden-layer MLP implemented with NumPy and backpropagation."""

    def __init__(self, input_dim, hidden_dim=32, learning_rate=0.03,
                 epochs=250, batch_size=32, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier-style initialization helps sigmoid networks start in a useful range.
        self.W1 = rng.normal(0, np.sqrt(1 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros((1, 1))
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = rng
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def bce(y, p):
        eps = 1e-8
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def predict_proba(self, X):
        h = self.sigmoid(X @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2).ravel()

    def fit(self, X_train, y_train, X_val, y_val):
        y_train = y_train.reshape(-1, 1).astype(float)
        y_val_col = y_val.reshape(-1, 1).astype(float)
        n = len(X_train)
        for _ in range(self.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb, yb = X_train[idx], y_train[idx]

                # Forward pass
                z1 = xb @ self.W1 + self.b1
                a1 = self.sigmoid(z1)
                z2 = a1 @ self.W2 + self.b2
                a2 = self.sigmoid(z2)

                # BCE + sigmoid output derivative simplifies to (prediction - target).
                dz2 = (a2 - yb) / len(xb)
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0, keepdims=True)
                dz1 = (dz2 @ self.W2.T) * a1 * (1 - a1)
                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Vanilla mini-batch gradient descent update
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            self.train_losses.append(self.bce(y_train, self.predict_proba(X_train).reshape(-1, 1)))
            self.val_losses.append(self.bce(y_val_col, self.predict_proba(X_val).reshape(-1, 1)))
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=["math", "portuguese"], default="math")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    try:
        X, y = load_student_data(args.subject)
        # Some ucimlrepo releases expose a combined dataset; prefer official course-specific file.
        if len(X) > 700 or len(X) < 300:
            X, y = load_student_data_csv(args.subject)
    except Exception:
        X, y = load_student_data_csv(args.subject)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )

    categorical = X_train_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X_train_raw.columns if c not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", encoder)]), categorical),
    ])
    X_train = prep.fit_transform(X_train_raw).astype(np.float64)
    X_val = prep.transform(X_val_raw).astype(np.float64)
    X_test = prep.transform(X_test_raw).astype(np.float64)

    # Scratch MLP: binary sigmoid + binary cross-entropy + mini-batch GD
    scratch = ScratchMLP(
        input_dim=X_train.shape[1], hidden_dim=32, learning_rate=0.03,
        epochs=args.epochs, batch_size=32, seed=RANDOM_STATE
    )
    start = time.perf_counter()
    scratch.fit(X_train, y_train, X_val, y_val)
    scratch_time = time.perf_counter() - start
    scratch_pred = (scratch.predict_proba(X_test) >= 0.5).astype(int)
    scratch_acc = accuracy_score(y_test, scratch_pred)

    # Library MLP. MLPClassifier uses logistic sigmoid for binary classification.
    # sklearn supports SGD or Adam; Adam is used here for robust convergence.
    library = MLPClassifier(
        hidden_layer_sizes=(32,), activation="logistic", solver="adam",
        learning_rate_init=0.003, max_iter=1, warm_start=True,
        batch_size=32, random_state=RANDOM_STATE, tol=0
    )
    lib_train_losses, lib_val_losses = [], []
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(args.epochs):
            library.fit(X_train, y_train)
            # sklearn's loss_ is its training objective (cross-entropy + small L2 term).
            lib_train_losses.append(float(library.loss_))
            p_val = np.clip(library.predict_proba(X_val)[:, 1], 1e-8, 1 - 1e-8)
            lib_val_losses.append(float(-np.mean(
                y_val * np.log(p_val) + (1 - y_val) * np.log(1 - p_val)
            )))
    library_time = time.perf_counter() - start
    lib_pred = library.predict(X_test)
    lib_acc = accuracy_score(y_test, lib_pred)

    print("\nDATASET: UCI Student Performance")
    print(f"Course: {'Mathematics' if args.subject == 'math' else 'Portuguese'}")
    print("Target: Pass (G3 >= 10) vs Fail (G3 < 10)")
    print(f"Samples: {len(y)} | Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")
    print("\nRESULTS")
    print(f"{'Model':<25}{'Test accuracy':>16}{'Training time (s)':>21}")
    print("-" * 62)
    print(f"{'Scratch NumPy MLP':<25}{scratch_acc:>15.4f}{scratch_time:>21.4f}")
    print(f"{'scikit-learn MLP':<25}{lib_acc:>15.4f}{library_time:>21.4f}")
    print("\nConfusion matrix labels: 0 = Fail, 1 = Pass")
    print("Scratch MLP:\n", confusion_matrix(y_test, scratch_pred, labels=[0, 1]))
    print("scikit-learn MLP:\n", confusion_matrix(y_test, lib_pred, labels=[0, 1]))

    epochs = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(epochs, scratch.train_losses, label="Scratch MLP")
    axes[0].plot(epochs, lib_train_losses, label="scikit-learn MLP")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy (library training loss includes L2)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, scratch.val_losses, label="Scratch MLP")
    axes[1].plot(epochs, lib_val_losses, label="scikit-learn MLP")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Learning Curves — Student Performance")
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=160, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, scratch_pred, labels=[0, 1], display_labels=["Fail", "Pass"],
        cmap="Blues", colorbar=False, ax=axes[0]
    )
    axes[0].set_title("Scratch NumPy MLP")
    ConfusionMatrixDisplay.from_predictions(
        y_test, lib_pred, labels=[0, 1], display_labels=["Fail", "Pass"],
        cmap="Greens", colorbar=False, ax=axes[1]
    )
    axes[1].set_title("scikit-learn MLP")
    fig.tight_layout()
    fig.savefig("confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.show()
    print("\nSaved plots: learning_curves.png, confusion_matrices.png")


if __name__ == "__main__":
    main()
