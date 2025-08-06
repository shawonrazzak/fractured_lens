import argparse, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# ───── helpers ────────────────────────────────────────────────────────
def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def angle_from_sincos(sin_, cos_): return np.arctan2(sin_.reshape(-1), cos_.reshape(-1))

# ───── data prep ──────────────────────────────────────────────────────
trajectory = "mirrorgames.csv"
TIME_STEPS = 4

sensor_inputs = ["r_x", "r_y",
                 "v_x_dot", "v_y_dot", "psi", "u_x", "u_y"]
wind_objectives = ["zeta_sin", "zeta_cos"]

df = pd.read_csv(trajectory)
df["zeta_sin"] = np.sin(df["zeta"])
df["zeta_cos"] = np.cos(df["zeta"])

def make_windows(f, w):
    X, Y = [], []
    for i in range(len(f) - w + 1):
        X.append(f.loc[i:i + w - 1, sensor_inputs].values)
        Y.append(f.loc[i + w - 1, wind_objectives].values)
    return np.stack(X), np.stack(Y)

X, Y = make_windows(df.reset_index(drop=True), TIME_STEPS)
num_feat = X.shape[-1]

# ───── model ─────────────────────────────────────────────────────────
def make_model(t, nf, no):
    i = tf.keras.Input((t, nf))
    x = layers.LSTM(64, return_sequences=True)(i)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    o = layers.Dense(no)(x)
    m = models.Model(i, o)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

# ───── plots ─────────────────────────────────────────────────────────
def loss_plot(h):
    plt.figure(dpi=100)
    plt.plot(h.history["loss"], label="train")
    if "val_loss" in h.history: plt.plot(h.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(); plt.show()

def parity(true_a, pred_a):
    plt.figure(dpi=120)
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k-')
    plt.scatter(true_a, pred_a, s=4, alpha=.3)
    plt.xlabel("true ζ (rad)"); plt.ylabel("pred ζ (rad)")
    plt.xlim(-np.pi, np.pi); plt.ylim(-np.pi, np.pi); plt.grid(); plt.show()

def err_hist(true_a, pred_a):
    e = wrap_to_pi(pred_a - true_a)
    plt.figure(dpi=120)
    plt.hist(np.rad2deg(e), bins=60, edgecolor='k', alpha=.6)
    plt.xlabel("ζ error (deg)"); plt.ylabel("count"); plt.show()

def table_fig(true_a, pred_a, rows=50):
    errors = wrap_to_pi(pred_a - true_a)
    abs_errors = np.abs(errors)
    total = len(true_a)

    worst_idx = np.argsort(-abs_errors)[:rows]
    best_idx = np.argsort(abs_errors)[:rows]

    spacing = max(1, total // (rows - 2))
    spaced_idx = np.arange(0, total, spacing)[:rows - 2]
    spaced_idx = np.unique(np.concatenate(([0], spaced_idx, [total - 1])))

    def build_table(indices, title):
        tbl = pd.DataFrame({
            "index": indices,
            "true ζ (deg)": np.rad2deg(true_a[indices]).round(2),
            "pred ζ (deg)": np.rad2deg(pred_a[indices]).round(2),
            "error (deg)": np.rad2deg(errors[indices]).round(2)
        })
        fig, ax = plt.subplots(figsize=(6, len(tbl)*0.22+1))
        ax.axis('off')
        t = ax.table(cellText=tbl.values,
                     colLabels=tbl.columns,
                     loc='center', cellLoc='center')
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
        plt.title(title, pad=10)
        plt.tight_layout(); plt.show()

    build_table(worst_idx, f"Top {rows} Worst Predictions (highest error)")
    build_table(best_idx,  f"Top {rows} Best Predictions (lowest error)")
    build_table(spaced_idx, f"{rows} Spaced Predictions (including start & end)")

# ───── train & evaluate ──────────────────────────────────────────────
MODEL_PATH = pathlib.Path("wind_rnn.keras")

def main(ep, batch):
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ loaded previous full model")
    else:
        model = make_model(TIME_STEPS, num_feat, Y.shape[-1])

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)

    cb = [callbacks.ModelCheckpoint(MODEL_PATH,
                                    save_best_only=False,
                                    save_weights_only=False,
                                    verbose=1)]

    hist = model.fit(X_train, Y_train, epochs=ep, batch_size=batch,
                     validation_data=(X_val, Y_val), shuffle=True,
                     callbacks=cb, verbose=1)

    # ── diagnostics ──
    pred_sin, pred_cos = model.predict(X, batch_size=batch).T
    true_sin, true_cos = Y.T
    true_ang = angle_from_sincos(true_sin, true_cos)
    pred_ang = angle_from_sincos(pred_sin, pred_cos)

    loss_plot(hist)
    parity(true_ang, pred_ang)
    err_hist(true_ang, pred_ang)
    table_fig(true_ang, pred_ang, rows=30)

# ───── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=256)
    args = ap.parse_args()
    main(args.epochs, args.batch)
