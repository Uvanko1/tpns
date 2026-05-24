import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os


def load_data(path: str, n_rows: int = 50_000) -> np.ndarray:
    df = pd.read_csv(
        path, sep=';', na_values='?',
        nrows=n_rows, low_memory=False
    )
    series = pd.to_numeric(df['Global_active_power'], errors='coerce').dropna().values.astype(np.float64)
    return series


def normalize(series: np.ndarray):
    mu, sigma = series.mean(), series.std()
    return (series - mu) / sigma, mu, sigma


def make_sequences(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)


def train_val_split(X, y, val_ratio=0.15):
    n = int(len(X) * (1 - val_ratio))
    return X[:n], y[:n], X[n:], y[n:]


def tanh(x):   return np.tanh(x)
def dtanh(x):  return 1.0 - np.tanh(x) ** 2
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def dsigmoid(x): s = sigmoid(x); return s * (1.0 - s)


def adam_step(params, grads, m, v, t,
              lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
    for name in grads:
        g = np.clip(grads[name], -clip, clip)
        m[name] = beta1 * m[name] + (1 - beta1) * g
        v[name] = beta2 * v[name] + (1 - beta2) * g ** 2
        m_hat = m[name] / (1 - beta1 ** t)
        v_hat = v[name] / (1 - beta2 ** t)
        params[name][:] -= lr * m_hat / (np.sqrt(v_hat) + eps)


class VanillaRNNCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = 1.0 / hidden_size ** 0.5
        self.Wx = rng.uniform(-k, k, (hidden_size, input_size))
        self.Wh = rng.uniform(-k, k, (hidden_size, hidden_size))
        self.bh = np.zeros(hidden_size)
        self.hidden_size = hidden_size
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ('Wx','Wh','bh')}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ('Wx','Wh','bh')}

    def params(self):  return {'Wx': self.Wx, 'Wh': self.Wh, 'bh': self.bh}
    def init_hidden(self): return np.zeros(self.hidden_size)

    def forward(self, x, h_prev):
        z = self.Wx @ x + self.Wh @ h_prev + self.bh
        h = tanh(z)
        return h, (x, h_prev, z)

    def backward(self, dh, cache):
        x, h_prev, z = cache
        dz = dh * dtanh(z)
        grads = dict(Wx=np.outer(dz, x), Wh=np.outer(dz, h_prev), bh=dz)
        dx      = self.Wx.T @ dz
        dh_prev = self.Wh.T @ dz
        return dx, dh_prev, grads


class GRUCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = 1.0 / hidden_size ** 0.5
        H, I = hidden_size, input_size
        def W(r, c): return rng.uniform(-k, k, (r, c))
        self.Wz = W(H, H + I); self.bz = np.full(H, -2.0)
        self.Wr = W(H, H + I); self.br = np.zeros(H)
        self.Wn = W(H, H + I); self.bn = np.zeros(H)
        self.hidden_size = H
        names = ('Wz','bz','Wr','br','Wn','bn')
        self.m = {n: np.zeros_like(getattr(self, n)) for n in names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in names}

    def params(self):
        return {n: getattr(self, n) for n in ('Wz','bz','Wr','br','Wn','bn')}

    def init_hidden(self): return np.zeros(self.hidden_size)

    def forward(self, x, h_prev):
        xh   = np.concatenate([h_prev, x])
        z_pre = self.Wz @ xh + self.bz;  z = sigmoid(z_pre)
        r_pre = self.Wr @ xh + self.br;  r = sigmoid(r_pre)
        xh_r  = np.concatenate([r * h_prev, x])
        n_pre = self.Wn @ xh_r + self.bn; n = tanh(n_pre)
        h = (1 - z) * h_prev + z * n
        cache = (x, h_prev, xh, z_pre, z, r_pre, r, xh_r, n_pre, n)
        return h, cache

    def backward(self, dh, cache):
        x, h_prev, xh, z_pre, z, r_pre, r, xh_r, n_pre, n = cache
        H = self.hidden_size

        dn      = dh * z
        dz      = dh * (n - h_prev)
        dh_prev = dh * (1 - z)

        dn_pre = dn * dtanh(n_pre)
        dxh_r  = self.Wn.T @ dn_pre
        dr_h   = dxh_r[:H];  dx_n = dxh_r[H:]
        dr     = dr_h * h_prev
        dh_prev += dr_h * r

        dr_pre   = dr * dsigmoid(r_pre)
        dxh_r_   = self.Wr.T @ dr_pre
        dh_prev += dxh_r_[:H]
        dx_r     = dxh_r_[H:]

        dz_pre   = dz * dsigmoid(z_pre)
        dxh_z    = self.Wz.T @ dz_pre
        dh_prev += dxh_z[:H]
        dx_z     = dxh_z[H:]

        dx = dx_z + dx_r + dx_n
        grads = dict(
            Wz=np.outer(dz_pre, xh),  bz=dz_pre,
            Wr=np.outer(dr_pre, xh),  br=dr_pre,
            Wn=np.outer(dn_pre, xh_r), bn=dn_pre,
        )
        return dx, dh_prev, grads


class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = 1.0 / hidden_size ** 0.5
        H, I = hidden_size, input_size
        self.W = rng.uniform(-k, k, (4 * H, H + I))
        self.b = np.zeros(4 * H)
        self.b[H:2*H] = 1.0          # forget gate bias = 1
        self.hidden_size = H
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ('W','b')}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ('W','b')}

    def params(self): return {'W': self.W, 'b': self.b}
    def init_hidden(self): return np.zeros(self.hidden_size), np.zeros(self.hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        H = self.hidden_size
        xh = np.concatenate([h_prev, x])
        g  = self.W @ xh + self.b
        i_pre, f_pre, g_pre, o_pre = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:]
        i, f, g_, o = sigmoid(i_pre), sigmoid(f_pre), tanh(g_pre), sigmoid(o_pre)
        c = f * c_prev + i * g_
        tanh_c = tanh(c)
        h = o * tanh_c
        cache = (xh, c_prev, i_pre, f_pre, g_pre, o_pre, i, f, g_, o, c, tanh_c)
        return h, c, cache

    def backward(self, dh, dc_next, cache):
        xh, c_prev, i_pre, f_pre, g_pre, o_pre, i, f, g_, o, c, tanh_c = cache
        H = self.hidden_size
        do = dh * tanh_c
        dc = dh * o * dtanh(c) + dc_next
        df, di, dg_, dc_prev = dc * c_prev, dc * g_, dc * i, dc * f
        dgates = np.concatenate([
            di * dsigmoid(i_pre),
            df * dsigmoid(f_pre),
            dg_ * dtanh(g_pre),
            do * dsigmoid(o_pre),
        ])
        dxh = self.W.T @ dgates
        grads = dict(W=np.outer(dgates, xh), b=dgates)
        return dxh[H:], dxh[:H], dc_prev, grads


class LinearLayer:
    def __init__(self, in_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = 1.0 / in_size ** 0.5
        self.W = rng.uniform(-k, k, (1, in_size))
        self.b = np.zeros(1)
        self.m = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.v = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

    def params(self): return {'W': self.W, 'b': self.b}

    def forward(self, h):
        return (self.W @ h + self.b)[0], h

    def backward(self, d_out, h):
        return self.W.T.squeeze() * d_out, {'W': d_out * h[np.newaxis, :], 'b': np.array([d_out])}


def _bptt_update(model_obj, d_pred, caches, lr):
    dh, out_grads = model_obj.out.backward(d_pred, caches['h_out'])
    model_obj.t += 1
    adam_step(model_obj.out.params(), out_grads,
              model_obj.out.m, model_obj.out.v, model_obj.t, lr)

    acc = {n: np.zeros_like(getattr(model_obj.cell, n))
           for n in model_obj.cell.params()}
    for c in reversed(caches['cell']):
        _, dh, grads = model_obj.cell.backward(dh, c)
        for n, g in grads.items():
            acc[n] += g

    adam_step(model_obj.cell.params(), acc,
              model_obj.cell.m, model_obj.cell.v, model_obj.t, lr)


def _bptt_update_lstm(model_obj, d_pred, caches, lr):
    dh, out_grads = model_obj.out.backward(d_pred, caches['h_out'])
    model_obj.t += 1
    adam_step(model_obj.out.params(), out_grads,
              model_obj.out.m, model_obj.out.v, model_obj.t, lr)

    acc = {n: np.zeros_like(getattr(model_obj.cell, n))
           for n in model_obj.cell.params()}
    dc = np.zeros_like(dh)
    for c in reversed(caches['cell']):
        _, dh, dc, grads = model_obj.cell.backward(dh, dc, c)
        for n, g in grads.items():
            acc[n] += g

    adam_step(model_obj.cell.params(), acc,
              model_obj.cell.m, model_obj.cell.v, model_obj.t, lr)


class ManualRNN:
    def __init__(self, input_size, hidden_size, seed=42):
        self.cell = VanillaRNNCell(input_size, hidden_size, seed)
        self.out  = LinearLayer(hidden_size, seed)
        self.t = 0

    def forward(self, seq):
        h = self.cell.init_hidden()
        cell_caches = []
        for x_t in seq:
            h, c = self.cell.forward(np.array([x_t]), h)
            cell_caches.append(c)
        pred, h_out = self.out.forward(h)
        return pred, {'cell': cell_caches, 'h_out': h_out}

    def backward_and_update(self, pred, target, caches, lr):
        loss = (pred - target) ** 2
        _bptt_update(self, 2.0 * (pred - target), caches, lr)
        return loss


class ManualGRU:
    def __init__(self, input_size, hidden_size, seed=42):
        self.cell = GRUCell(input_size, hidden_size, seed)
        self.out  = LinearLayer(hidden_size, seed)
        self.t = 0

    def forward(self, seq):
        h = self.cell.init_hidden()
        cell_caches = []
        for x_t in seq:
            h, c = self.cell.forward(np.array([x_t]), h)
            cell_caches.append(c)
        pred, h_out = self.out.forward(h)
        return pred, {'cell': cell_caches, 'h_out': h_out}

    def backward_and_update(self, pred, target, caches, lr):
        loss = (pred - target) ** 2
        _bptt_update(self, 2.0 * (pred - target), caches, lr)
        return loss


class ManualLSTM:
    def __init__(self, input_size, hidden_size, seed=42):
        self.cell = LSTMCell(input_size, hidden_size, seed)
        self.out  = LinearLayer(hidden_size, seed)
        self.t = 0

    def forward(self, seq):
        h, c = self.cell.init_hidden()
        cell_caches = []
        for x_t in seq:
            h, c, cache = self.cell.forward(np.array([x_t]), (h, c))
            cell_caches.append(cache)
        pred, h_out = self.out.forward(h)
        return pred, {'cell': cell_caches, 'h_out': h_out}

    def backward_and_update(self, pred, target, caches, lr):
        loss = (pred - target) ** 2
        _bptt_update_lstm(self, 2.0 * (pred - target), caches, lr)
        return loss


def train(model, X_train, y_train, X_val, y_val,
          epochs=5, lr=5e-4, report_every=500):
    train_losses, val_losses = [], []
    n = len(X_train)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = np.random.permutation(n)
        ep_loss = 0.0
        for step, i in enumerate(idx):
            pred, caches = model.forward(X_train[i])
            loss = model.backward_and_update(pred, y_train[i], caches, lr)
            ep_loss += loss
            if (step + 1) % report_every == 0:
                print(f"  ep{ep} step {step+1}/{n}  train_mse={ep_loss/(step+1):.4f}", flush=True)
        ep_loss /= n
        train_losses.append(ep_loss)
        val_preds = np.array([model.forward(X_val[i])[0] for i in range(len(X_val))])
        val_mse = np.mean((val_preds - y_val) ** 2)
        val_mae = np.mean(np.abs(val_preds - y_val))
        val_losses.append(val_mse)
        dt = time.time() - t0
        print(f"Epoch {ep}/{epochs}  train_mse={ep_loss:.4f}  "
              f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  time={dt:.1f}s")
    return train_losses, val_losses


def evaluate(model, X_test, y_test, mu, sigma):
    preds_norm = np.array([model.forward(X_test[i])[0] for i in range(len(X_test))])
    preds = preds_norm * sigma + mu
    true  = y_test  * sigma + mu
    rmse  = np.sqrt(np.mean((preds - true) ** 2))
    mae   = np.mean(np.abs(preds - true))
    mape  = np.mean(np.abs((preds - true) / (np.abs(true) + 1e-8))) * 100
    return rmse, mae, mape, preds, true


def plot_results(results, true_vals, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle('Ручная реализация RNN / GRU / LSTM\nПрогноз энергопотребления',
                 fontsize=13, fontweight='bold')
    ax = axes[0]
    for name, (tl, vl, *_) in results.items():
        ep = range(1, len(tl) + 1)
        ax.plot(ep, tl,  label=f'{name} train', linewidth=2)
        ax.plot(ep, vl, '--', label=f'{name} val',   linewidth=1.5)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('MSE (норм.)')
    ax.set_title('Динамика потерь (нормализованные данные)')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    n_show = 300
    ax.plot(true_vals[:n_show], 'k-', label='Факт', linewidth=1.5, alpha=0.8)
    colors = {'RNN': '#e74c3c', 'GRU': '#2ecc71', 'LSTM': '#3498db'}
    for name, (*_, preds, _) in results.items():
        ax.plot(preds[:n_show], '--', label=name,
                color=colors.get(name), linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Шаг'); ax.set_ylabel('Global Active Power (кВт)')
    ax.set_title(f'Прогноз vs факт (первые {n_show} точек тестовой выборки)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[График сохранён: {out_path}]")


def print_table(results):
    print("\n" + "═" * 52)
    print(f"{'Модель':<8} {'RMSE (кВт)':>12} {'MAE (кВт)':>12} {'MAPE (%)':>10}")
    print("─" * 52)
    for name, (_, _, rmse, mae, mape, *_) in results.items():
        print(f"{name:<8} {rmse:>12.4f} {mae:>12.4f} {mape:>10.2f}")
    print("═" * 52)


if __name__ == '__main__':
    DATA_PATH = 'household_power_consumption.txt'
    N_ROWS  = 5_000
    SEQ_LEN = 32
    HIDDEN  = 24
    EPOCHS  = 3
    LR      = 1e-3

    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)

    print("═" * 55)
    print("  РУЧНАЯ РЕАЛИЗАЦИЯ RNN / GRU / LSTM")
    print("═" * 55)

    print("\n[1] Загрузка и подготовка данных...")
    series = load_data(DATA_PATH, n_rows=N_ROWS)
    print(f"    Длина ряда: {len(series):,} точек")

    series_norm, mu, sigma = normalize(series)
    X, y = make_sequences(series_norm, SEQ_LEN)
    X_tr, y_tr, X_tmp, y_tmp = train_val_split(X, y, 0.2)
    X_val, y_val, X_te, y_te = train_val_split(X_tmp, y_tmp, 0.5)
    print(f"    Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_te)}")

    models = {
        'RNN':  ManualRNN(1, HIDDEN),
        'GRU':  ManualGRU(1, HIDDEN),
        'LSTM': ManualLSTM(1, HIDDEN),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'─'*45}")
        print(f"  Обучение {name}  (hidden={HIDDEN}, epochs={EPOCHS})")
        print(f"{'─'*45}")
        tl, vl = train(model, X_tr, y_tr, X_val, y_val,
                       epochs=EPOCHS, lr=LR, report_every=500)
        rmse, mae, mape, preds, true = evaluate(model, X_te, y_te, mu, sigma)
        results[name] = (tl, vl, rmse, mae, mape, preds, true)
        print(f"  Тест → RMSE={rmse:.4f} кВт  MAE={mae:.4f} кВт  MAPE={mape:.2f}%")

    print_table(results)
    true_vals = results['RNN'][6]
    plot_results(results, true_vals,
                 os.path.join(OUT_DIR, 'manual_rnn_results.png'))
    print("\nГотово! Файл: manual_rnn_results.png")