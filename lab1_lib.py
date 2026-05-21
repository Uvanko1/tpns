import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from chefboost import Chefboost as chef

df = pd.read_csv("iris.data")


print("\nПропуски:")
print(df.isnull().sum())

print("\nКоличество дубликатов:")
print(df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

x = df.drop(columns="class")

corr_matr = x.corr()
print(corr_matr.astype(float))

plt.figure(figsize=(8,6))
sns.heatmap(corr_matr, annot=True, cmap="coolwarm")
plt.title("Матрица корреляций")
plt.show()

threshold = 0.9
corr_matrix_abs = corr_matr.abs()

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

new_df = df_reduced.rename(columns={"class": "Decision"})
new_df["Decision"] = new_df["Decision"].astype("object")
config = {
    "algorithm": "C4.5"
}

model = chef.fit(new_df, config)

print("\nВажность признаков:")
print(chef.feature_importance("outputs/rules/rules.py"))
