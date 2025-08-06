import argparse, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
trajectory_file = "seen.csv"
time_steps      = 4
model_path      = "wind_direction_rnn_model.keras"

sensor_inputs     = ["r_x", "r_y", "v_x_dot", "v_y_dot", "psi", "u_x", "u_y"]
estimator_outputs = ["zeta_sin", "zeta_cos"]

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_from_sincos_deg(sin_, cos_):
    return np.degrees(np.arctan2(sin_.reshape(-1), cos_.reshape(-1)))

def form_frames(df, t, in_cols, out_cols, overlap=False):
    inputs, targets = [], []
    step = 1 if overlap else t
    for i in range(0, len(df) - t + 1, step):
        window = df.iloc[i:i+t]
        if len(window) == t:
            inputs.append(window[in_cols].values)
            targets.append(window.iloc[-1][out_cols].values)
    return np.array(inputs), np.array(targets)

def create_lstm_model(t, in_dim, out_dim=2):
    inp = Input(shape=(t, in_dim))
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.LSTM(64)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(out_dim)(x)
    return models.Model(inp, out)

# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------
def loss_plot(h):
    plt.figure(dpi=100)
    plt.plot(h.history["loss"], label="train")
    if "val_loss" in h.history:
        plt.plot(h.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(); plt.show()

def parity_plot(true_a, pred_a):
    plt.figure(dpi=120)
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], "k--")
    plt.scatter(true_a, pred_a, s=4, alpha=0.3)
    plt.xlabel("True ζ (rad)"); plt.ylabel("Predicted ζ (rad)")
    plt.xlim(-np.pi, np.pi); plt.ylim(-np.pi, np.pi); plt.grid(); plt.show()

def err_hist(true_a, pred_a):
    err = wrap_to_pi(pred_a - true_a)
    plt.figure(dpi=120)
    plt.hist(np.rad2deg(err), bins=60, edgecolor="k", alpha=0.6)
    plt.xlabel("ζ Error (deg)"); plt.ylabel("Count"); plt.grid(); plt.show()

def table_fig(true_a, pred_a, rows=30):
    err  = wrap_to_pi(pred_a - true_a)
    aerr = np.abs(err)
    worst = np.argsort(-aerr)[:rows]
    best  = np.argsort(aerr)[:rows]
    spacing = max(1, len(true_a) // (rows - 2))
    spaced = np.arange(0, len(true_a), spacing)[:rows - 2]
    spaced = np.unique(np.concatenate(([0], spaced, [len(true_a) - 1])))

    def _tbl(idx, title):
        df = pd.DataFrame({
            "idx": idx,
            "true (deg)":  np.rad2deg(true_a[idx]).round(2),
            "pred (deg)":  np.rad2deg(pred_a[idx]).round(2),
            "error (deg)": np.rad2deg(err[idx]).round(2)
        })
        fig, ax = plt.subplots(figsize=(6, len(df)*0.23 + 1))
        ax.axis("off")
        t = ax.table(cellText=df.values, colLabels=df.columns,
                     loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
        plt.title(title, pad=10); plt.tight_layout(); plt.show()

    _tbl(worst,  f"Top {rows} Worst")
    _tbl(best,   f"Top {rows} Best")
    _tbl(spaced, f"{rows} Evenly Spaced")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(epochs, batch_size):
    # Load data
    df = pd.read_csv(trajectory_file)
    df["zeta_sin"] = np.sin(df["zeta"])
    df["zeta_cos"] = np.cos(df["zeta"])

    X, Y = form_frames(df, time_steps, sensor_inputs, estimator_outputs)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    # Load or create model
    if pathlib.Path(model_path).exists():
        print("Loaded saved model.")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating a new model.")
        model = create_lstm_model(time_steps, X.shape[-1])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        min_delta=1e-4,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[early_stop]
    )

    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save model architecture diagram
    plot_model(model, to_file="ann_structure.png", show_shapes=True, show_layer_names=True)
    print("ANN structure saved to ann_structure.png")

    # Evaluation
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE : {mae:.4f}")

    preds      = model.predict(X_test, batch_size=batch_size)
    pred_ang   = np.deg2rad(angle_from_sincos_deg(preds[:, 0], preds[:, 1]))
    true_ang   = np.deg2rad(angle_from_sincos_deg(Y_test[:, 0], Y_test[:, 1]))

    # Diagnostics
    loss_plot(history)
    parity_plot(true_ang, pred_ang)
    err_hist(true_ang, pred_ang)
    table_fig(true_ang, pred_ang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs for each run")
    parser.add_argument("--batch",  type=int, default=32, help="Batch size")
    args = parser.parse_args()
    main(args.epochs, args.batch)
