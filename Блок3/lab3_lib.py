import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

CFG = dict(
    data_path   = 'household_power_consumption.txt',
    n_rows      = 50000,
    seq_len     = 24,
    batch_size  = 512,
    hidden_size = 32,
    num_layers  = 2,
    dropout     = 0.2,
    lr          = 3e-3,
    epochs      = 5,
    patience    = 3,
    val_ratio   = 0.15,
    test_ratio  = 0.10,
    seed        = 42,
    out_dir     = 'outputs',
)

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {DEVICE}")



def load_series(path: str, n_rows: int) -> np.ndarray:
    df = pd.read_csv(
        path, sep=';', na_values='?',
        nrows=n_rows, low_memory=False,
    )
    series = pd.to_numeric(df['Global_active_power'], errors='coerce').interpolate().values.astype(np.float32)
    return series


class PowerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def prepare_data(cfg):
    series = load_series(cfg['data_path'], cfg['n_rows'])
    print(f"Длина ряда: {len(series):,}  |  min={series.min():.2f}  max={series.max():.2f}")

    mu, sigma = series.mean(), series.std()
    norm = (series - mu) / sigma

    L = cfg['seq_len']
    X = np.lib.stride_tricks.sliding_window_view(norm[:-1], L)
    y = norm[L:]

    n = len(y)
    n_test = int(n * cfg['test_ratio'])
    n_val  = int(n * cfg['val_ratio'])
    n_tr   = n - n_val - n_test

    ds_tr  = PowerDataset(X[:n_tr], y[:n_tr])
    ds_val = PowerDataset(X[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val])
    ds_te  = PowerDataset(X[n_tr+n_val:], y[n_tr+n_val:])

    print(f"Train={len(ds_tr):,}  Val={len(ds_val):,}  Test={len(ds_te):,}")

    bs = cfg['batch_size']
    loaders = dict(
        train = DataLoader(ds_tr,  batch_size=bs, shuffle=True,  drop_last=True, num_workers=0),
        val   = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=0),
        test  = DataLoader(ds_te,  batch_size=bs, shuffle=False, num_workers=0),
    )
    return loaders, mu, sigma


class RecurrentModel(nn.Module):

    def __init__(self, cell_type: str, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.cell_type = cell_type.upper()
        kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(**kwargs, nonlinearity='tanh')
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(**kwargs)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(**kwargs)
        else:
            raise ValueError(f"Неизвестный тип ячейки: {cell_type}")

        self.dropout_out = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                if 'bias_hh' in name and self.cell_type == 'LSTM':
                    n = p.numel() // 4
                    p.data[n:2*n].fill_(1.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = self.dropout_out(out[:, -1, :])
        return self.fc(last)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


class EarlyStopping:
    def __init__(self, patience=5, delta=1e-5):
        self.patience = patience
        self.delta = delta
        self.best = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_model(cell_type: str, loaders: dict, cfg: dict):
    model = RecurrentModel(
        cell_type   = cell_type,
        input_size  = 1,
        hidden_size = cfg['hidden_size'],
        num_layers  = cfg['num_layers'],
        dropout     = cfg['dropout'],
    ).to(DEVICE)

    print(f"\n{'─'*50}")
    print(f"  {cell_type}  параметры: {model.n_params:,}")
    print(f"{'─'*50}")

    optimizer  = Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion  = nn.MSELoss()
    stopper    = EarlyStopping(patience=cfg['patience'])

    history = {'train': [], 'val': []}
    t0 = time.time()

    for ep in range(1, cfg['epochs'] + 1):
        tr_loss  = run_epoch(model, loaders['train'], optimizer, criterion, train=True)
        val_loss = run_epoch(model, loaders['val'],   optimizer, criterion, train=False)
        scheduler.step(val_loss)

        history['train'].append(tr_loss)
        history['val'].append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  ep{ep:>2}/{cfg['epochs']}  "
              f"train={tr_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.6f}")

        if stopper.step(val_loss, model):
            print(f"  → Early stopping на эпохе {ep}")
            break

    stopper.restore(model)
    print(f"  Время обучения: {time.time()-t0:.1f}s")
    return model, history


def evaluate_model(model, loader, mu, sigma):
    model.eval()
    preds_n, trues_n = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds_n.append(model(X_b.to(DEVICE)).cpu().numpy())
            trues_n.append(y_b.numpy())
    preds_n = np.concatenate(preds_n).flatten()
    trues_n = np.concatenate(trues_n).flatten()

    preds = preds_n * sigma + mu
    trues = trues_n * sigma + mu

    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    mae  = np.mean(np.abs(preds - trues))
    mape = np.mean(np.abs((preds - trues) / (np.abs(trues) + 1e-8))) * 100
    r2   = 1 - np.sum((trues - preds)**2) / (np.sum((trues - trues.mean())**2) + 1e-8)
    return dict(rmse=rmse, mae=mae, mape=mape, r2=r2, preds=preds, trues=trues)



def plot_all(all_history, all_metrics, out_path):
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Библиотечная реализация PyTorch: RNN / GRU / LSTM\n'
                 'Прогноз энергопотребления (Global_active_power)',
                 fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    colors = {'RNN': '#e74c3c', 'GRU': '#27ae60', 'LSTM': '#2980b9'}

    for idx, name in enumerate(['RNN', 'GRU', 'LSTM']):
        ax = fig.add_subplot(gs[idx, 0])
        h = all_history[name]
        ep = range(1, len(h['train']) + 1)
        ax.plot(ep, h['train'], label='Train', color=colors[name], linewidth=2)
        ax.plot(ep, h['val'],   label='Val',   color=colors[name],
                linewidth=1.5, linestyle='--')
        ax.set_title(f'{name} — потери MSE (нормализованные)')
        ax.set_xlabel('Эпоха'); ax.set_ylabel('MSE')
        ax.legend(); ax.grid(alpha=0.3)

    ax_pred = fig.add_subplot(gs[:, 1])
    n_show = 500
    first = True
    for name, metrics in all_metrics.items():
        if first:
            ax_pred.plot(metrics['trues'][:n_show], 'k-',
                         label='Факт', linewidth=1.2, alpha=0.8)
            first = False
        ax_pred.plot(metrics['preds'][:n_show], '--',
                     label=f'{name} (RMSE={metrics["rmse"]:.3f})',
                     color=colors[name], linewidth=1.4, alpha=0.85)

    ax_pred.set_title(f'Прогноз vs Факт — первые {n_show} точек теста')
    ax_pred.set_xlabel('Шаг'); ax_pred.set_ylabel('Global Active Power (кВт)')
    ax_pred.legend(fontsize=9); ax_pred.grid(alpha=0.3)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"[График сохранён: {out_path}]")


def plot_bar_comparison(all_metrics, out_path):
    names   = list(all_metrics.keys())
    rmses   = [all_metrics[n]['rmse']  for n in names]
    maes    = [all_metrics[n]['mae']   for n in names]
    mapes   = [all_metrics[n]['mape']  for n in names]
    r2s     = [all_metrics[n]['r2']    for n in names]

    x = np.arange(len(names))
    width = 0.22
    colors = ['#e74c3c', '#27ae60', '#2980b9']

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle('Сравнение метрик на тестовой выборке', fontsize=13, fontweight='bold')

    for ax, vals, title, ylabel in zip(
        axes,
        [rmses, maes, mapes, r2s],
        ['RMSE (кВт)', 'MAE (кВт)', 'MAPE (%)', 'R²'],
        ['кВт', 'кВт', '%', '']
    ):
        bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
        ax.set_title(title); ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"[График сохранён: {out_path}]")


def print_summary(all_metrics):
    print("\n" + "═" * 65)
    print(f"{'Модель':<8} {'RMSE (кВт)':>12} {'MAE (кВт)':>12} {'MAPE (%)':>10} {'R²':>8}")
    print("─" * 65)
    for name, m in all_metrics.items():
        print(f"{name:<8} {m['rmse']:>12.4f} {m['mae']:>12.4f} "
              f"{m['mape']:>10.2f} {m['r2']:>8.4f}")
    print("═" * 65)
    best = min(all_metrics, key=lambda n: all_metrics[n]['rmse'])
    print(f"\nЛучшая модель по RMSE: {best}")



if __name__ == '__main__':
    os.makedirs(CFG['out_dir'], exist_ok=True)

    print("═" * 55)
    print("  БИБЛИОТЕЧНАЯ РЕАЛИЗАЦИЯ (PyTorch): RNN / GRU / LSTM")
    print("═" * 55)

    print("\n[1] Загрузка данных...")
    loaders, mu, sigma = prepare_data(CFG)

    print("\n[2] Обучение моделей...")
    all_history = {}
    all_metrics = {}

    for cell in ['RNN', 'GRU', 'LSTM']:
        model, history = train_model(cell, loaders, CFG)
        metrics = evaluate_model(model, loaders['test'], mu, sigma)
        all_history[cell] = history
        all_metrics[cell] = metrics
        print(f"  Тест [{cell}]: RMSE={metrics['rmse']:.4f}  "
              f"MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%  R²={metrics['r2']:.4f}")

    print("\n[3] Итоговая таблица:")
    print_summary(all_metrics)

    print("\n[4] Сохранение графиков...")
    plot_all(all_history, all_metrics,
             os.path.join(CFG['out_dir'], 'torch_rnn_curves.png'))
    plot_bar_comparison(all_metrics,
                        os.path.join(CFG['out_dir'], 'torch_rnn_bars.png'))

    print("\nГотово! Файлы: torch_rnn_curves.png, torch_rnn_bars.png")
