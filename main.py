import os
from pathlib import Path
from typing import Tuple

import keras_tuner as kt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostRegressor
from category_encoders import CatBoostEncoder
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
matplotlib.use('TkAgg')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PREPROC_DIR = ROOT / "preprocessors"
TUNER_DIR = ROOT / "tuner_dir"


def main():
    current_year = 2026
    df = load_single_file(DATASET_DIR / f"QualiData_{current_year}_cleaned.csv")

    train_catboost_models(df)
    train_laptime_model(df)


# def r2_score(y_true, y_pred):
#     SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
#     SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
#     return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


@tf.keras.utils.register_keras_serializable(package="metrics", name="R2Score")
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.total_SS_res = self.add_weight(name='total_SS_res', initializer='zeros')
        self.total_SS_tot = self.add_weight(name='total_SS_tot', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.total_SS_res.assign_add(SS_res)
        self.total_SS_tot.assign_add(SS_tot)

    def result(self):
        return 1 - self.total_SS_res / (self.total_SS_tot + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_SS_res.assign(0)
        self.total_SS_tot.assign(0)


@tf.keras.utils.register_keras_serializable(package="metrics", name="huber_loss")
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.reduce_mean(tf.where(is_small_error, squared_loss, linear_loss))


# Обучение CatBoost-моделей
def train_catboost_models(df: pd.DataFrame) -> Tuple:
    """
    Обучает CatBoost-модели для предсказания AirTemp, TrackTemp, Pressure.
    """
    # AirTemp
    at_features = [
        "Year",
        "Humidity",
        "Pressure",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "CircuitId",
        "Event",
        "TrackTemp",
    ]

    at_numeric_features = [
        "Year",
        "Humidity",
        "Pressure",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "TrackTemp",
    ]

    at_cat_features = ["CircuitId"]
    at_oh_features = ["Event"]
    at_target = "AirTemp"

    at_X = df[at_features]
    at_y = df[at_target].to_frame()

    at_x_train, at_x_test, at_y_train, at_y_test = train_test_split(
        at_X, at_y, test_size=0.1, random_state=42
    )
    at_x_train, at_x_val, at_y_train, at_y_val = train_test_split(
        at_x_train, at_y_train, test_size=0.33, random_state=42
    )

    at_preprocessor = ColumnTransformer(
        transformers=[
            ("cat_enc", CatBoostEncoder(sigma=0.1, a=5), at_cat_features),
            ("oh_enc", OneHotEncoder(handle_unknown="ignore"), at_oh_features),
            ("r_sc", RobustScaler(), at_numeric_features),
        ],
        remainder="drop",
    )

    at_y_scaler = RobustScaler()
    at_preprocessor.fit(at_x_train, at_y_train)

    at_y_train_scaled = at_y_scaler.fit_transform(at_y_train.values)
    at_X_train_processed = at_preprocessor.transform(at_x_train)

    at_cb_model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
    at_cb_model.fit(at_X_train_processed, at_y_train_scaled)
    at_cb_model.save_model(MODEL_DIR / "at_cat_boost_model.cbm")

    dump(at_preprocessor, PREPROC_DIR / "at_data_preprocessor.joblib")
    dump(at_y_scaler, PREPROC_DIR / "at_y_scaler.joblib")

    # TrackTemp
    tt_features = [
        "Year",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "CircuitId",
        "Event",
    ]

    tt_numeric_features = [
        "Year",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
    ]

    tt_cat_features = ["CircuitId"]
    tt_oh_features = ["Event"]

    tt_target = "TrackTemp"

    tt_X = df[tt_features]
    tt_y = df[tt_target].to_frame()

    tt_x_train, tt_x_test, tt_y_train, tt_y_test = train_test_split(
        tt_X, tt_y, test_size=0.1, random_state=42
    )
    tt_x_train, tt_x_val, tt_y_train, tt_y_val = train_test_split(
        tt_x_train, tt_y_train, test_size=0.33, random_state=42
    )

    tt_preprocessor = ColumnTransformer(
        transformers=[
            ("cat_enc", CatBoostEncoder(sigma=0.1, a=5), tt_cat_features),
            ("oh_enc", OneHotEncoder(handle_unknown="ignore"), tt_oh_features),
            ("r_sc", RobustScaler(), tt_numeric_features),
        ],
        remainder="drop",
    )

    tt_y_scaler = RobustScaler()
    tt_preprocessor.fit(tt_x_train, tt_y_train)

    tt_y_train_scaled = tt_y_scaler.fit_transform(tt_y_train.values)
    tt_X_train_processed = tt_preprocessor.transform(tt_x_train)

    tt_cb_model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
    tt_cb_model.fit(tt_X_train_processed, tt_y_train_scaled)
    tt_cb_model.save_model(MODEL_DIR / "tt_cat_boost_model.cbm")

    dump(tt_preprocessor, PREPROC_DIR / "tt_data_preprocessor.joblib")
    dump(tt_y_scaler, PREPROC_DIR / "tt_y_scaler.joblib")

    # Pressure
    pres_features = [
        "Year",
        "Humidity",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "CircuitId",
        "Event",
        "AirTemp",
        "TrackTemp",
    ]

    pres_numeric_features = [
        "Year",
        "Humidity",
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "AirTemp",
        "TrackTemp",
    ]

    pres_cat_features = ["CircuitId"]
    pres_oh_features = ["Event"]

    pres_target = "Pressure"

    pres_X = df[pres_features]
    pres_y = df[pres_target].to_frame()

    pres_x_train, pres_x_test, pres_y_train, pres_y_test = train_test_split(
        pres_X, pres_y, test_size=0.1, random_state=42
    )
    pres_x_train, pres_x_val, pres_y_train, pres_y_val = train_test_split(
        pres_x_train, pres_y_train, test_size=0.33, random_state=42
    )

    pres_preprocessor = ColumnTransformer(
        transformers=[
            ("cat_enc", CatBoostEncoder(sigma=0.1, a=5), pres_cat_features),
            ("oh_enc", OneHotEncoder(handle_unknown="ignore"), pres_oh_features),
            ("r_sc", RobustScaler(), pres_numeric_features),
        ],
        remainder="drop",
    )

    pres_y_scaler = RobustScaler()
    pres_preprocessor.fit(pres_x_train, pres_y_train)

    pres_y_train_scaled = pres_y_scaler.fit_transform(pres_y_train.values)
    pres_X_train_processed = pres_preprocessor.transform(pres_x_train)

    pres_cb_model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
    pres_cb_model.fit(pres_X_train_processed, pres_y_train_scaled)
    pres_cb_model.save_model(MODEL_DIR / "pres_cat_boost_model.cbm")

    dump(pres_preprocessor, PREPROC_DIR / "pres_data_preprocessor.joblib")
    dump(pres_y_scaler, PREPROC_DIR / "pres_y_scaler.joblib")

    return (
        (at_cb_model, at_preprocessor, at_y_scaler),
        (tt_cb_model, tt_preprocessor, tt_y_scaler),
        (pres_cb_model, pres_preprocessor, pres_y_scaler),
    )


# Обучение модели
def train_laptime_model(df: pd.DataFrame) -> None:
    """
    Обучает CatBoost + нейросеть для предсказания LapTime.
    """

    def set_seed(seed: int = 42) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    set_seed(42)

    df = df.dropna(subset=["LapTime"])

    df["IsStreetCircuit"] = df["IsStreetCircuit"].astype(int)
    df["Rainfall"] = df["Rainfall"].astype(int)
    df["IsHighHumidity"] = df["IsHighHumidity"].astype(int)

    features = [
        "Driver",
        "Team",
        "Year",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "DriverPoints",
        "TeamPoints",
        "DriverAvgQualiPos",
        "TeamAvgQualiPos",
        "IsStreetCircuit",
        "F1Era",
        "CircuitCorners",
        "TrackAirDiff",
        "IsHighHumidity",
        "DriverQualiPace",
        "DriverSeasons",
        "CircuitLength",
        "TeamPointsContribution",
        "CarEngine",
        "Event",
        "CircuitId",
    ]

    target = "LapTime"

    X = df[features]
    y = df[target].to_frame()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.333, random_state=42
    )

    print(f"Всего: {len(X)}")
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    oh_enc_col = ["F1Era", "CarEngine", "Event"]
    cat_enc_col_new = ["Driver", "Team", "CircuitId"]
    numeric_features = [
        "Rainfall",
        "IsStreetCircuit",
        "IsHighHumidity",
        "Year",
        "AirTemp",
        "Humidity",
        "Pressure",
        "TrackTemp",
        "DriverPoints",
        "TeamPoints",
        "DriverAvgQualiPos",
        "TeamAvgQualiPos",
        "CircuitCorners",
        "TrackAirDiff",
        "DriverQualiPace",
        "DriverSeasons",
        "CircuitLength",
        "TeamPointsContribution",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("r_sc", RobustScaler(), numeric_features),
            ("cat_enc", CatBoostEncoder(sigma=0.1, a=5), cat_enc_col_new),
            ("oh_enc", OneHotEncoder(handle_unknown="ignore"), oh_enc_col),
        ]
    )

    y_scaler = RobustScaler()
    preprocessor.fit(x_train, y_train)

    y_train_scaled = y_scaler.fit_transform(y_train.values)
    y_val_scaled = y_scaler.transform(y_val.values)
    y_test_scaled = y_scaler.transform(y_test.values)

    X_train_processed = preprocessor.transform(x_train)
    X_val_processed = preprocessor.transform(x_val)
    X_test_processed = preprocessor.transform(x_test)

    dump(preprocessor, PREPROC_DIR / "data_preprocessor.joblib")
    dump(y_scaler, PREPROC_DIR / "y_scaler.joblib")

    delta_orig = 0.2
    delta_scaled = delta_orig / y_scaler.scale_[0]

    # CatBoost
    cb_model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
    cb_model.fit(X_train_processed, y_train_scaled)
    cb_model.save_model(MODEL_DIR / "cat_boost_model.cbm")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train_processed.shape[0], 1))

    for train_idx, val_idx in kf.split(X_train_processed):
        X_tr, X_val_fold = X_train_processed[train_idx], X_train_processed[val_idx]
        y_tr, y_val_fold = y_train_scaled[train_idx], y_train_scaled[val_idx]

        model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val_fold).reshape(-1, 1)

    rf_train_preds = oof_preds

    final_cb_model = CatBoostRegressor(loss_function="Huber:delta=1.0", depth=6, verbose=0)
    final_cb_model.fit(X_train_processed, y_train_scaled)

    rf_val_preds = final_cb_model.predict(X_val_processed).reshape(-1, 1)
    rf_test_preds = final_cb_model.predict(X_test_processed).reshape(-1, 1)

    X_train_catnn = np.hstack([X_train_processed, rf_train_preds])
    X_val_catnn = np.hstack([X_val_processed, rf_val_preds])
    X_test_catnn = np.hstack([X_test_processed, rf_test_preds])

    # Нейросеть с keras-tuner
    def new_model_builder(hp) -> tf.keras.Model:
        model = tf.keras.Sequential()

        input_dim = X_train_catnn.shape[1]

        hp_units1 = hp.Int("units1", 32, 512, step=32)
        hp_units2 = hp.Int("units2", 16, 256, step=16)
        hp_dropout = hp.Float("dropout", 0.15, 0.3, step=0.05)

        reg_type = hp.Choice("reg_type", ["l1", "l2", "l1l2"])
        reg_factor = hp.Float("reg_factor", 1e-4, 1e-3, sampling="log")
        lr = hp.Choice("learning_rate", [3e-3, 1e-3, 5e-4])
        activation = hp.Choice("activation", ["leaky_relu", "relu", "swish"])
        use_bn = hp.Boolean("batch_norm")

        if reg_type == "l1":
            reg = tf.keras.regularizers.L1(reg_factor)
        elif reg_type == "l2":
            reg = tf.keras.regularizers.L2(reg_factor)
        else:
            reg = tf.keras.regularizers.L1L2(l1=reg_factor, l2=reg_factor)

        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        model.add(tf.keras.layers.Dense(hp_units1, kernel_regularizer=reg))

        if activation == "leaky_relu":
            model.add(tf.keras.layers.LeakyReLU())
        else:
            model.add(tf.keras.layers.Activation(activation))

        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(hp_dropout))
        model.add(tf.keras.layers.Dense(hp_units2, kernel_regularizer=reg))

        if activation == "leaky_relu":
            model.add(tf.keras.layers.LeakyReLU())
        else:
            model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Dense(1, activation="linear"))

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
            loss=lambda y_true, y_pred: huber_loss(y_true, y_pred, delta_scaled),
            metrics=["mse", "mae", R2Score()],
        )

        return model

    tuner = kt.BayesianOptimization(
        new_model_builder,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=300,
        executions_per_trial=1,
        directory=TUNER_DIR,
        project_name="f1_laptime",
    )

    def get_callbacks():
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=15,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.3,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "best_checkpoint.keras",
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        )
        return [early_stop, reduce_lr, checkpoint]

    tuner.search(
        X_train_catnn,
        y_train_scaled,
        epochs=500,
        validation_data=(X_val_catnn, y_val_scaled),
        batch_size=64,
        callbacks=get_callbacks(),
        verbose=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(MODEL_DIR / "best_laptime_model.keras")

    # Оценка на тесте
    print("\nNeuro model:")
    test_loss, test_mse, test_mae, test_r_squared = best_model.evaluate(
        X_test_catnn, y_test_scaled, verbose=1
    )
    print(
        f"Test Huber: {test_loss:.6f}, Test MAE: {test_mae:.6f}, "
        f"Test R²: {test_r_squared:.6f}, Test MSE: {test_mse:.6f}"
    )

    y_pred_neuro = best_model.predict(X_test_catnn)
    r2 = r2_score(y_test_scaled, y_pred_neuro)
    print(f"R² Score (sklearn): {r2:.6f}")

    y_test_inversed = y_scaler.inverse_transform(y_test_scaled)
    y_pred_inversed = y_scaler.inverse_transform(y_pred_neuro)

    mse_error = mean_squared_error(y_test_inversed, y_pred_inversed)
    mae_error = mean_absolute_error(y_test_inversed, y_pred_inversed)
    mape_error = mean_absolute_percentage_error(y_test_inversed, y_pred_inversed)

    print(f"MSE: {mse_error:.6f}, MAE: {mae_error:.6f}, MAPE: {mape_error:.6f}")

    # CatBoost оценка
    print("\nCatBoostRegressor baseline:")
    cb_y_pred = cb_model.predict(X_test_processed).reshape(-1, 1)
    y_pred_cat_inversed = y_scaler.inverse_transform(cb_y_pred)

    r2_cat = r2_score(y_test_inversed, y_pred_cat_inversed)
    print(f"R² Score: {r2_cat:.6f}")

    mse_cat = mean_squared_error(y_test_inversed, y_pred_cat_inversed)
    mae_cat = mean_absolute_error(y_test_inversed, y_pred_cat_inversed)
    mape_cat = mean_absolute_percentage_error(y_test_inversed, y_pred_cat_inversed)

    print(f"MSE: {mse_cat:.6f}, MAE: {mae_cat:.6f}, MAPE: {mape_cat:.6f}")

    # Визуализация
    plot_predictions_comparison(y_test_inversed, y_pred_inversed, y_pred_cat_inversed)
    plot_model_architecture(best_model, MODEL_DIR / "model_architecture.png")

def plot_predictions_comparison(
        y_true: np.ndarray,
        y_pred_neuro: np.ndarray,
        y_pred_cat: np.ndarray,) -> None:
    """
    Строит графики сравнения предсказаний нейросети и CatBoost.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred_neuro, alpha=0.5, label="Нейросеть", color="blue")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "k--",
        lw=2,
        label="Идеальное предсказание",
    )
    plt.xlabel("Реальное время круга (сек)")
    plt.ylabel("Спрогнозированное время круга (сек)")
    plt.title("Нейросеть: Предсказания vs Реальные значения")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    axes[0].scatter(
        y_true, y_pred_neuro, alpha=0.5, color="blue", label="Нейросеть"
    )
    axes[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "k--",
        lw=2,
    )
    axes[0].set_xlabel("Реальное время круга (сек)")
    axes[0].set_ylabel("Предсказанное время круга (сек)")
    axes[0].set_title("Нейросеть")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(
        y_true, y_pred_cat, alpha=0.5, color="red", label="CatBoost"
    )
    axes[1].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "k--",
        lw=2,
    )
    axes[1].set_xlabel("Реальное время круга (сек)")
    axes[1].set_title("CatBoost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Сравнение моделей", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_model_architecture(model: tf.keras.Model, save_path: Path) -> None:
    """
    Сохраняет архитектуру модели в виде PNG.
    """
    tf.keras.utils.plot_model(
        model,
        to_file=save_path,
        show_shapes=True,
        show_layer_names=True,
    )
    print(f"Архитектура модели сохранена: {save_path}")

def load_single_file(file_path: str) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    encodings = ("utf-8-sig", "utf-8", "cp1251", "latin1", "windows-1252")
    separators = (",", ";", "\t")

    errors = []

    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=separator,
                    encoding=encoding,
                    engine="python",
                    skipinitialspace=True,
                    quotechar='"',
                    on_bad_lines="warn",
                )

                if not df.empty:
                    return df

            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                errors.append(f"encoding={encoding}, sep={separator!r}: {error}")
                continue

    raise RuntimeError(
        f"Не удалось прочитать файл {file_path}."
    )


if __name__ == '__main__':
    main()
