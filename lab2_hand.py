import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("iris.data")
df = df.drop_duplicates()

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

X = df.drop(columns=["class"]).values
y = df["class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

num_classes = len(np.unique(y))
y_onehot = np.zeros((y.size, num_classes))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


class MLP:
    def __init__(self, input_size, hidden1, hidden2, output_size, lr=0.1):
        self.lr = lr

        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(1 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(1 / hidden2)
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)

        return self.a3

    def loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dz3 = y_pred - y_true
        dW3 = self.a2.T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ self.W3.T) * sigmoid_deriv(self.z2)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * sigmoid_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y)

            self.backward(X, y, y_pred)

            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)



model = MLP(
    input_size=X.shape[1],
    hidden1=10,
    hidden2=10,
    output_size=num_classes,
    lr=0.1
)

model.fit(X_train, y_train, epochs=1644)


y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred == y_true)
print("\nAccuracy:", accuracy)