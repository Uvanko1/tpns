import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("  MLP КЛАССИФИКАТОР — Iris")
print("=" * 55)

df_iris = pd.read_csv("iris.data")

print("Пропуски:\n", df_iris.isnull().sum())
df_iris = df_iris.drop_duplicates()

le = LabelEncoder()
df_iris["class"] = le.fit_transform(df_iris["class"])

X_iris = df_iris.drop(columns=["class"])
y_iris = df_iris["class"]

corr = X_iris.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X_iris = X_iris.drop(columns=to_drop)
print("Удалённые признаки (Iris):", to_drop)

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_iris_scaled, y_iris, test_size=0.2, random_state=42
)

clf = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="logistic",
    solver="adam",
    max_iter=10000,
    random_state=42,
    verbose=True,
)
clf.fit(X_train_c, y_train_c)

y_pred_c = clf.predict(X_test_c)

mae  = mean_absolute_error(y_test_c, y_pred_c)
rmse = np.sqrt(mean_squared_error(y_test_c, y_pred_c))
r2   = r2_score(y_test_c, y_pred_c)

print(f"\nМетрики на тесте (ирисы):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

print("\nПервые 10 предсказаний vs реальность:")
print(f"{'Предсказание':>15}  {'Реальность':>12}  {'Ошибка':>10}")
for pred, true in zip(y_pred_c[:10], y_test_c.values[:10]):
    print(f"{pred:15.2f}  {true:12.2f}  {abs(pred - true):10.2f}")
print(f"\nТочность на тесте (Iris): {accuracy_score(y_test_c, y_pred_c):.4f}\n")


print("=" * 55)
print("  MLP РЕГРЕССОР — Цена ноутбука")
print("=" * 55)

df_lap = pd.read_csv("Laptop_price.csv")

print("Пропуски:\n", df_lap.isnull().sum())
df_lap = df_lap.drop_duplicates()

le_brand = LabelEncoder()
df_lap["Brand"] = le_brand.fit_transform(df_lap["Brand"])

feature_cols = ["Brand", "Processor_Speed", "RAM_Size",
                "Storage_Capacity", "Screen_Size", "Weight"]
target_col = "Price"

X_lap = df_lap[feature_cols]
y_lap = df_lap[target_col]

corr_lap = X_lap.corr().abs()
upper_lap = corr_lap.where(np.triu(np.ones(corr_lap.shape), k=1).astype(bool))
to_drop_lap = [col for col in upper_lap.columns if any(upper_lap[col] > 0.9)]
X_lap = X_lap.drop(columns=to_drop_lap)
print("Удалённые признаки (ноутбуки):", to_drop_lap)

scaler_lap = StandardScaler()
X_lap_scaled = scaler_lap.fit_transform(X_lap)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_lap_scaled, y_lap, test_size=0.2, random_state=42
)

reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=10000,
    random_state=42,
    verbose=True,
)
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)

mae  = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2   = r2_score(y_test_r, y_pred_r)

print(f"\nМетрики на тесте (цена ноутбука):")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")

print("\nПервые 10 предсказаний vs реальность:")
print(f"{'Предсказание':>15}  {'Реальность':>12}  {'Ошибка':>10}")
for pred, true in zip(y_pred_r[:10], y_test_r.values[:10]):
    print(f"{pred:15.2f}  {true:12.2f}  {abs(pred - true):10.2f}")