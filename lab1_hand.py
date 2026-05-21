import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.data")

print("\nПропуски:")
print(df.isnull().sum())

print("\nКоличество дубликатов:")
print(df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

df_c = df.drop(columns=["class"])
print(df_c.head())

cols = df_c.columns
x = df_c
corr_matrix = pd.DataFrame(index=cols, columns=cols)
for i in df_c.columns:
    for j in df_c.columns:
        mean_x = np.mean(df_c[i])
        mean_y = np.mean(df_c[j])
        S_xy = np.sum((df_c[i] - mean_x) * (df_c[j] - mean_y))
        S_x = np.sqrt(np.sum((df_c[i] - mean_x) ** 2))
        S_y = np.sqrt(np.sum((df_c[j] - mean_y) ** 2))
        r = S_xy / (S_x * S_y)
        corr_matrix.loc[i, j] = r
print(corr_matrix.astype(float))

corr_matrix = corr_matrix.astype(float)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Матрица корреляций")
plt.show()

threshold = 0.9
corr_matrix_abs = corr_matrix.abs()

upper_triangle = corr_matrix_abs.where(
    np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool)
)

to_drop = [
    column for column in upper_triangle.columns
    if any(upper_triangle[column] > threshold)
]

print("\nСильно коррелирующие признаки для удаления:")
print(to_drop)

df_reduced = df.drop(columns=to_drop)
print(df_reduced.head())

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()

    ent = 0
    for p in probabilities:
        if p > 0:
            ent -= p * np.log2(p)
    return ent



def gain_ratio_numeric(X, y, feature):
    total_entropy = entropy(y)

    sorted_indices = np.argsort(X[feature].values)
    X_sorted = X.iloc[sorted_indices]
    y_sorted = y[sorted_indices]

    unique_values = np.unique(X_sorted[feature])

    best_gr = 0
    best_threshold = None

    for i in range(1, len(unique_values)):
        threshold = (unique_values[i - 1] + unique_values[i]) / 2

        left_mask = X_sorted[feature] <= threshold
        right_mask = X_sorted[feature] > threshold

        y_left = y_sorted[left_mask]
        y_right = y_sorted[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        weighted_entropy = (
                weight_left * entropy(y_left) +
                weight_right * entropy(y_right)
        )

        info_gain = total_entropy - weighted_entropy

        split_info = 0
        for w in [weight_left, weight_right]:
            if w > 0:
                split_info -= w * np.log2(w)

        if split_info == 0:
            continue

        gr = info_gain / split_info

        if gr > best_gr:
            best_gr = gr
            best_threshold = threshold

    return best_gr, best_threshold



def compute_gain_ratio(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    results = {}

    for feature in X.columns:
        gr, threshold = gain_ratio_numeric(X, y, feature)
        results[feature] = {
            "gain_ratio": gr,
            "best_threshold": threshold
        }

    return results
results = compute_gain_ratio(df_reduced, "class")

for feature, values in results.items():
    print(f"\nПризнак: {feature}")
    print(f"Gain Ratio: {values['gain_ratio']:.4f}")
print("\n")