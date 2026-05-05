import numpy as np


class SoftMarginSVM:
    def __init__(self, lr=0.001, epochs=20, C=1.0, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.random_state = random_state
        self.W = None
        self.b = None
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        self.W = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.losses = []

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for ith in indices:
                x_i = X[ith]
                y_i = y[ith]

                score = np.dot(x_i, self.W) + self.b
                margin = y_i * score

                if margin >= 1:
                    dW = self.W
                    db = 0.0
                else:
                    dW = self.W - self.C * y_i * x_i
                    db = -self.C * y_i

                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db

            epoch_loss = self.loss_fn(X, y)
            self.losses.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

    def decision_function(self, X: np.ndarray):
        return X @ self.W + self.b

    def predict(self, X: np.ndarray):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def loss_fn(self, X: np.ndarray, y: np.ndarray):
        scores = self.decision_function(X)
        hinge_losses = np.maximum(0, 1 - y * scores)

        regularization_loss = 0.5 * np.dot(self.W, self.W)
        hinge_loss = self.C * np.sum(hinge_losses)

        return regularization_loss + hinge_loss

    def get_metrics(self, X: np.ndarray, y: np.ndarray, positive_label=-1):
        y_pred = self.predict(X)

        TP = np.sum((y == positive_label) & (y_pred == positive_label))
        FP = np.sum((y != positive_label) & (y_pred == positive_label))
        FN = np.sum((y == positive_label) & (y_pred != positive_label))
        TN = np.sum((y != positive_label) & (y_pred != positive_label))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }