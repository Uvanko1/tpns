import numpy as np
import struct, gzip, os, time
from pathlib import Path


RNG = np.random.default_rng(42)

def _load_mnist_keras():
    from tensorflow import keras
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
    return x_tr, y_tr, x_te, y_te


def preprocess(x_train, y_train, x_test, y_test):
    def pad(x):
        return np.pad(x, ((0,0),(2,2),(2,2)), mode='constant')
    x_train = pad(x_train)[..., np.newaxis].astype(np.float32) / 255.0
    x_test  = pad(x_test )[..., np.newaxis].astype(np.float32) / 255.0
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)


def tanh(x):    return np.tanh(x)
def tanh_d(x):  return 1.0 - np.tanh(x)**2
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def xavier(shape):
    fan_in  = int(np.prod(shape[1:]))
    fan_out = shape[0]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return RNG.uniform(-lim, lim, shape).astype(np.float32)


def im2col(x, kh, kw, stride=1):
    N, H, W, C = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    col = np.zeros((N, out_h, out_w, kh * kw * C), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :]
            col[:, i, j, :] = patch.reshape(N, -1)
    return col


def col2im(dcol, x_shape, kh, kw, stride=1):
    N, H, W, C = x_shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    dx = np.zeros(x_shape, dtype=dcol.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = dcol[:, i, j, :].reshape(N, kh, kw, C)
            dx[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :] += patch
    return dx


def conv_forward(x, W, b):
    C_out, kh, kw, C_in = W.shape
    col = im2col(x, kh, kw)
    N, oh, ow, _ = col.shape
    W_flat = W.reshape(C_out, -1)
    out = col.reshape(-1, kh*kw*C_in) @ W_flat.T + b
    out = out.reshape(N, oh, ow, C_out)
    return out, (x, col, W, b)


def conv_backward(dout, cache):
    x, col, W, b = cache
    C_out, kh, kw, C_in = W.shape
    N, oh, ow, _ = col.shape

    db = dout.sum(axis=(0,1,2))

    dout_flat = dout.reshape(-1, C_out)
    col_flat  = col.reshape(-1, kh*kw*C_in)
    dW_flat   = dout_flat.T @ col_flat
    dW = dW_flat.reshape(W.shape)

    W_flat  = W.reshape(C_out, -1)
    dcol_flat = dout_flat @ W_flat
    dcol = dcol_flat.reshape(N, oh, ow, kh*kw*C_in)
    dx = col2im(dcol, x.shape, kh, kw)

    return dx, dW, db


def avgpool_forward(x, pool=2):
    N, H, W, C = x.shape
    oh, ow = H//pool, W//pool
    out = x.reshape(N, oh, pool, ow, pool, C).mean(axis=(2,4))
    return out, x.shape
def avgpool_backward(dout, cache, pool=2):
    x_shape = cache
    N, H, W, C = x_shape
    oh, ow = H//pool, W//pool
    dx = (dout / (pool*pool))[:, :, np.newaxis, :, np.newaxis, :]
    dx = np.broadcast_to(dx, (N, oh, pool, ow, pool, C)).copy()
    return dx.reshape(x_shape)

def dense_forward(x, W, b):
    out = x @ W + b
    return out, (x, W, b)

def dense_backward(dout, cache):
    x, W, b = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db


def cross_entropy_loss(logits, labels):
    probs = softmax(logits)
    N = logits.shape[0]
    log_p = -np.log(probs[np.arange(N), labels] + 1e-9)
    loss = log_p.mean()
    dlogits = probs
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits


class Params:
    def __init__(self):
        self.Wc1 = xavier((6,  5, 5, 1));
        self.bc1 = np.zeros(6, dtype=np.float32)
        self.Wc3 = xavier((16, 5, 5, 6));
        self.bc3 = np.zeros(16, dtype=np.float32)
        self.Wc5 = xavier((120,5, 5,16));
        self.bc5 = np.zeros(120, dtype=np.float32)
        self.Wf6 = xavier((120, 84));
        self.bf6 = np.zeros(84, dtype=np.float32)
        self.Wo  = xavier((84, 10));
        self.bo  = np.zeros(10, dtype=np.float32)
        self._mom = {k: np.zeros_like(v) for k, v in self.__dict__.items()}

    def update(self, grads, lr, momentum=0.9):
        for k, g in grads.items():
            self._mom[k] = momentum * self._mom[k] + g
            setattr(self, k, getattr(self, k) - lr * self._mom[k])


def forward(x, p):
    caches = {}

    z1, caches['c1'] = conv_forward(x,p.Wc1, p.bc1)
    a1 = tanh(z1)
    caches['z1'] = z1
    s2, caches['s2'] = avgpool_forward(a1)

    z3, caches['c3'] = conv_forward(s2,p.Wc3, p.bc3)
    a3 = tanh(z3)
    caches['z3'] = z3
    s4, caches['s4'] = avgpool_forward(a3)

    z5, caches['c5'] = conv_forward(s4,p.Wc5, p.bc5)
    a5 = tanh(z5)
    caches['z5'] = z5
    flat = a5.reshape(a5.shape[0], -1)
    caches['flat_shape'] = a5.shape

    z6, caches['f6'] = dense_forward(flat,p.Wf6, p.bf6)
    a6 = tanh(z6)
    caches['z6'] = z6

    zo, caches['fo'] = dense_forward(a6,p.Wo,  p.bo)

    return zo, caches


def backward(dzo, caches, p):
    grads = {}

    da6, grads['Wo'], grads['bo'] = dense_backward(dzo, caches['fo'])

    dz6 = da6 * tanh_d(caches['z6'])
    dflat, grads['Wf6'], grads['bf6'] = dense_backward(dz6, caches['f6'])

    da5 = dflat.reshape(caches['flat_shape'])
    dz5 = da5 * tanh_d(caches['z5'])

    ds4, grads['Wc5'], grads['bc5'] = conv_backward(dz5, caches['c5'])

    da3 = avgpool_backward(ds4, caches['s4'])
    dz3 = da3 * tanh_d(caches['z3'])

    ds2, grads['Wc3'], grads['bc3'] = conv_backward(dz3, caches['c3'])

    da1 = avgpool_backward(ds2, caches['s2'])
    dz1 = da1 * tanh_d(caches['z1'])

    _, grads['Wc1'], grads['bc1'] = conv_backward(dz1, caches['c1'])

    return grads


def accuracy(logits, labels):
    return (logits.argmax(axis=1) == labels).mean()


def train(x_train, y_train, x_test, y_test,
          epochs=5, batch_size=64, lr=0.01, momentum=0.9):

    p = Params()
    N = x_train.shape[0]
    history = {"train_loss":[], "train_acc":[], "test_acc":[]}

    for epoch in range(1, epochs+1):
        t0 = time.time()
        idx = RNG.permutation(N)
        x_train, y_train = x_train[idx], y_train[idx]

        epoch_loss, epoch_acc = 0.0, 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            xb = x_train[start:start+batch_size]
            yb = y_train[start:start+batch_size]

            logits, caches = forward(xb, p)
            loss, dlogits  = cross_entropy_loss(logits, yb)
            grads          = backward(dlogits, caches, p)
            p.update(grads, lr, momentum)

            epoch_loss += loss
            epoch_acc  += accuracy(logits, yb)
            n_batches  += 1

            if n_batches % 50 == 0:
                print(f"  batch {n_batches:4d}/{N//batch_size}  "
                      f"loss={loss:.4f}", end="\r")

        test_logits, _ = forward(x_test[:1000], p)
        test_acc = accuracy(test_logits, y_test[:1000])

        history["train_loss"].append(epoch_loss / n_batches)
        history["train_acc" ].append(epoch_acc  / n_batches)
        history["test_acc"  ].append(test_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs}  "
              f"loss={epoch_loss/n_batches:.4f}  "
              f"train_acc={epoch_acc/n_batches*100:.1f}%  "
              f"test_acc={test_acc*100:.1f}%  "
              f"({elapsed:.1f}s)")

    return p, history




if __name__ == "__main__":
    print("Loading MNIST …")
    x_tr, y_tr, x_te, y_te = _load_mnist_keras()
    x_tr, y_tr, x_te, y_te = preprocess(x_tr, y_tr, x_te, y_te)
    print(f"Train: {x_tr.shape}  |  Test: {x_te.shape}")

    SUBSET = 10_000
    x_tr, y_tr = x_tr[:SUBSET], y_tr[:SUBSET]
    print(f"(Using {SUBSET} training samples for speed demonstration)")

    trained_params, history = train(
        x_tr, y_tr, x_te, y_te,
        epochs=5,
        batch_size=64,
        lr=0.01,
        momentum=0.9,
    )

    print("\nEvaluating on full 10 000 test images …")
    EVAL_BATCH = 500
    all_preds = []
    for start in range(0, x_te.shape[0], EVAL_BATCH):
        logits, _ = forward(x_te[start:start+EVAL_BATCH], trained_params)
        all_preds.append(logits.argmax(axis=1))
    all_preds = np.concatenate(all_preds)
    final_acc = (all_preds == y_te).mean()
    print(f"Final test accuracy: {final_acc*100:.2f}%")

    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_te, all_preds)
    print("\nConfusion matrix (строки = истинный класс, столбцы = предсказанный):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_te, all_preds))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs_x = range(1, len(history["train_loss"])+1)
        axes[0].plot(epochs_x, history["train_loss"], marker='o', label="Train loss")
        axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[1].plot(epochs_x, [v*100 for v in history["train_acc"]], marker='o', label="Train")
        axes[1].plot(epochs_x, [v*100 for v in history["test_acc" ]], marker='s', label="Test (1k)")
        axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
        plt.suptitle(f"LeNet-5 NumPy — Final test acc: {final_acc*100:.2f}%")
        plt.tight_layout()
        plt.savefig("lenet5_numpy_curves.png", dpi=120)
        plt.show()
        print("Saved lenet5_numpy_curves.png")

        fig3, ax3 = plt.subplots(figsize=(8, 7))
        im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax3)
        ax3.set_xticks(range(10));
        ax3.set_yticks(range(10))
        ax3.set_xlabel("Predicted label");
        ax3.set_ylabel("True label")
        ax3.set_title(f"Confusion Matrix — LeNet-5 NumPy\nTest acc: {final_acc * 100:.2f}%")
        thresh = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax3.text(j, i, str(cm[i, j]),
                         ha='center', va='center', fontsize=8,
                         color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig("lenet5_numpy_confusion_matrix.png", dpi=120)
        plt.show()
        print("Saved lenet5_numpy_confusion_matrix.png")

        sample_logits, _ = forward(x_te[:16], trained_params)
        preds_vis = sample_logits.argmax(axis=1)

        fig2, axes2 = plt.subplots(2, 8, figsize=(14, 4))
        for i, ax in enumerate(axes2.flat):
            img = x_te[i, 2:30, 2:30, 0]
            ax.imshow(img, cmap="gray")
            colour = "green" if preds_vis[i] == y_te[i] else "red"
            ax.set_title(f"P:{preds_vis[i]} T:{y_te[i]}", color=colour, fontsize=8)
            ax.axis("off")
        plt.suptitle("LeNet-5 NumPy predictions (green=correct, red=wrong)")
        plt.tight_layout()
        plt.savefig("lenet5_numpy_predictions.png", dpi=120)
        plt.show()
        print("Saved lenet5_numpy_predictions.png")

    except ImportError:
        pass

