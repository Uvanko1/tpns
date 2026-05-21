import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.data")

print("Пропуски:\n", df.isnull().sum())

df = df.drop_duplicates()

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

X = df.drop(columns=["class"])
y = df["class"]

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X = X.drop(columns=to_drop)

print("Удалённые признаки:", to_drop)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='logistic',
    solver='adam',
    max_iter=10000,
    random_state=42,
    verbose=True
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))