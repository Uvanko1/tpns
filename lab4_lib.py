import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), mode='constant')
x_test  = np.pad(x_test,  ((0,0),(2,2),(2,2)), mode='constant')

x_train = x_train[..., np.newaxis].astype("float32") / 255.0
x_test  = x_test [..., np.newaxis].astype("float32") / 255.0

print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")


def build_lenet5(activation: str = "tanh") -> keras.Model:
    model = keras.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation=activation,
                      input_shape=(32, 32, 1), padding="valid", name="C1"),
        layers.AveragePooling2D(pool_size=2, strides=2, name="S2"),

        layers.Conv2D(16, kernel_size=5, strides=1, activation=activation,
                      padding="valid", name="C3"),
        layers.AveragePooling2D(pool_size=2, strides=2, name="S4"),

        layers.Conv2D(120, kernel_size=5, strides=1, activation=activation,
                      padding="valid", name="C5"),

        layers.Flatten(),

        layers.Dense(84, activation=activation, name="F6"),

        layers.Dense(10, activation="softmax", name="Output"),
    ], name="LeNet-5")
    return model


model = build_lenet5(activation="tanh")
model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1,
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc*100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")

all_preds = np.argmax(model.predict(x_test), axis=1)

print("\nConfusion matrix (строки = истинный класс, столбцы = предсказанный):")
cm = confusion_matrix(y_test, all_preds)
print(cm)
print("\nClassification report:")
print(classification_report(y_test, all_preds))

fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm).plot(ax=ax_cm, colorbar=True, cmap="Blues")
ax_cm.set_title(f"Confusion Matrix — LeNet-5 Keras\nTest acc: {test_acc * 100:.2f}%")
plt.tight_layout()
plt.savefig("lenet5_keras_confusion_matrix.png", dpi=120)
plt.show()
print("Saved lenet5_keras_confusion_matrix.png")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy");
ax1.set_xlabel("Epoch");
ax1.legend()

ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss");
ax2.set_xlabel("Epoch");
ax2.legend()

plt.suptitle(f"LeNet-5 (Keras) — Test acc: {test_acc * 100:.2f}%")
plt.tight_layout()
plt.savefig("lenet5_keras_curves.png", dpi=120)
plt.show()

preds = np.argmax(model.predict(x_test[:16]), axis=1)
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    colour = "green" if preds[i] == y_test[i] else "red"
    ax.set_title(f"P:{preds[i]} T:{y_test[i]}", color=colour, fontsize=8)
    ax.axis("off")
plt.suptitle("LeNet-5 predictions (green=correct, red=wrong)")
plt.tight_layout()
plt.savefig("lenet5_keras_predictions.png", dpi=120)
plt.show()
