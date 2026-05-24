import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


class MLPClassifier:
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
            if epoch % 100 == 0:
                print(f"[Classifier] Epoch {epoch:4d}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)



class MLPRegressor:

    def __init__(self, input_size, hidden1, hidden2, lr=0.01):
        self.lr = lr

        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(1 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(1 / hidden2)
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dz3 = 2 * (y_pred - y_true) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        dz2 = (dz3 @ self.W3.T) * sigmoid_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ self.W2.T) * sigmoid_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=2000):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y)
            self.backward(X, y, y_pred)
            if epoch % 200 == 0:
                print(f"[Regressor]  Epoch {epoch:4d}, MSE: {loss:.4f}")

    def predict(self, X):
        return self.forward(X).ravel()

print("=" * 55)
print("  MLP КЛАССИФИКАТОР — Iris")
print("=" * 55)

df_iris = pd.read_csv("iris.data")
df_iris = df_iris.drop_duplicates()

le = LabelEncoder()
df_iris["class"] = le.fit_transform(df_iris["class"])

X_iris = df_iris.drop(columns=["class"]).values
y_iris = df_iris["class"].values

scaler_iris = StandardScaler()
X_iris = scaler_iris.fit_transform(X_iris)

num_classes = len(np.unique(y_iris))
y_onehot = np.zeros((y_iris.size, num_classes))
y_onehot[np.arange(y_iris.size), y_iris] = 1

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_iris, y_onehot, test_size=0.2, random_state=42
)

clf = MLPClassifier(
    input_size=X_iris.shape[1],
    hidden1=10,
    hidden2=10,
    output_size=num_classes,
    lr=0.1,
)
clf.fit(X_train_c, y_train_c, epochs=1644)

y_pred_c = clf.predict(X_test_c)
y_true_c = np.argmax(y_test_c, axis=1)
accuracy = np.mean(y_pred_c == y_true_c)
print(f"\nТочность на тесте (Iris): {accuracy:.4f}\n")



print("=" * 55)
print("  MLP РЕГРЕССОР — Цена ноутбука")
print("=" * 55)

df_lap = pd.read_csv("Laptop_price.csv")
df_lap = df_lap.drop_duplicates()

le_brand = LabelEncoder()
df_lap["Brand"] = le_brand.fit_transform(df_lap["Brand"])

feature_cols = ["Brand", "Processor_Speed", "RAM_Size",
                "Storage_Capacity", "Screen_Size", "Weight"]
target_col = "Price"

X_lap = df_lap[feature_cols].values
y_lap = df_lap[target_col].values

scaler_X = StandardScaler()
X_lap = scaler_X.fit_transform(X_lap)

scaler_y = StandardScaler()
y_lap_scaled = scaler_y.fit_transform(y_lap.reshape(-1, 1)).ravel()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_lap, y_lap_scaled, test_size=0.2, random_state=42
)

reg = MLPRegressor(
    input_size=X_lap.shape[1],
    hidden1=32,
    hidden2=16,
    lr=0.01,
)
reg.fit(X_train_r, y_train_r, epochs=2000)

y_pred_r_scaled = reg.predict(X_test_r)
y_pred_r = scaler_y.inverse_transform(y_pred_r_scaled.reshape(-1, 1)).ravel()
y_true_r = scaler_y.inverse_transform(y_test_r.reshape(-1, 1)).ravel()

mae  = np.mean(np.abs(y_pred_r - y_true_r))
rmse = np.sqrt(np.mean((y_pred_r - y_true_r) ** 2))
ss_res = np.sum((y_true_r - y_pred_r) ** 2)
ss_tot = np.sum((y_true_r - np.mean(y_true_r)) ** 2)
r2   = 1 - ss_res / ss_tot

print(f"\nМетрики на тесте (цена ноутбука):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

print("\nПервые 10 предсказаний vs реальность:")
print(f"{'Предсказание':>15}  {'Реальность':>12}  {'Ошибка':>10}")
for pred, true in zip(y_pred_r[:10], y_true_r[:10]):
    print(f"{pred:15.2f}  {true:12.2f}  {abs(pred - true):10.2f}")