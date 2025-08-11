import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score as r2score
import pandas as pd
import os
import random
from joblib import dump, load

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import keras_tuner as kt

matplotlib.use('TkAgg')

import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from catboost import CatBoostRegressor
from category_encoders import CatBoostEncoder


def main():
    df = load_single_file("C:/Users/PC/Desktop/f1_py/dataset/QualiData_cleaned.csv")
    neuro3(df)


def r2_score(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


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


def neuro3(df):
    def set_seed(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    set_seed(42)

    df = df.dropna(subset=['LapTime'])

    df["IsStreetCircuit"] = df["IsStreetCircuit"].astype(int)
    df["Rainfall"] = df["Rainfall"].astype(int)
    df["IsHighHumidity"] = df["IsHighHumidity"].astype(int)

    features = ['Driver', 'Team', 'Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
                'DriverPoints', 'TeamPoints', 'DriverAvgQualiPos', 'TeamAvgQualiPos', 'IsStreetCircuit', 'F1Era',
                'CircuitCorners', 'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons',
                'CircuitLength', 'TeamPointsContribution', 'CarEngine', 'Event']

    target = 'LapTime'

    X = df[features]
    y = df[target].to_frame()

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.333, random_state=42
    )

    print(len(X))
    print(f"X_train:{len(x_train)}")
    print(f"X_val:{len(x_val)}")
    print(f"X_test:{len(x_test)}")

    oh_enc_col = ['F1Era', 'CarEngine', 'Event']

    cat_enc_col_new = ['Driver', 'Team']

    numeric_features = ['Rainfall', 'IsStreetCircuit', 'IsHighHumidity', 'Year', 'AirTemp', 'Humidity', 'Pressure',
                        'TrackTemp', 'DriverPoints', 'TeamPoints', 'DriverAvgQualiPos', 'TeamAvgQualiPos',
                        'CircuitCorners', 'TrackAirDiff', 'DriverQualiPace', 'DriverSeasons', 'CircuitLength',
                        'TeamPointsContribution']

    # Настройки нормализации
    preprocessor = ColumnTransformer(
        transformers=[
            ('r_sc', RobustScaler(), numeric_features),
            ('cat_enc', CatBoostEncoder(sigma=0.1, a=5), cat_enc_col_new),
            ('oh_enc', OneHotEncoder(handle_unknown='ignore'), oh_enc_col)
        ])

    y_scaler = RobustScaler()
    preprocessor.fit(x_train, y_train)

    y_train_scaled = y_scaler.fit_transform(y_train.values)
    y_val_scaled = y_scaler.transform(y_val.values)
    y_test_scaled = y_scaler.transform(y_test.values)

    X_train_processed = preprocessor.transform(x_train)
    X_val_processed = preprocessor.transform(x_val)
    X_test_processed = preprocessor.transform(x_test)

    cb_feature_names = preprocessor.get_feature_names_out().tolist()

    dump(preprocessor, 'preprocessors/data_preprocessor.joblib')
    dump(y_scaler, 'preprocessors/y_scaler.joblib')

    # cb_model = CatBoostRegressor(loss_function='Huber:delta=1.0', depth=6, verbose=1)
    # cb_model.fit(X_train_processed, y_train_scaled)
    # cb_model.save_model("models/cat_boost_model.cbm")

    cb_model = CatBoostRegressor().load_model("models/cat_boost_model.cbm")

    tt_features = ['Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'IsStreetCircuit', 'CircuitCorners',
                   'IsHighHumidity', 'CircuitLength']

    tt_target = 'TrackTemp'

    tt_X = df[tt_features]
    tt_y = df[tt_target].to_frame()

    tt_x_train, tt_x_test, tt_y_train, tt_y_test = train_test_split(
        tt_X, tt_y, test_size=0.1, random_state=42
    )

    tt_x_train, tt_x_val, tt_y_train, tt_y_val = train_test_split(
        tt_x_train, tt_y_train, test_size=0.33, random_state=42
    )

    tt_numeric_features = ['Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'IsStreetCircuit',
                           'CircuitCorners', 'IsHighHumidity', 'CircuitLength']

    tt_preprocessor = ColumnTransformer(
        transformers=[
            ('r_sc', RobustScaler(), tt_numeric_features)
        ])

    tt_y_scaler = RobustScaler()
    tt_preprocessor.fit(tt_x_train, tt_y_train)

    tt_y_train_scaled = tt_y_scaler.fit_transform(tt_y_train.values)

    tt_X_train_processed = tt_preprocessor.transform(tt_x_train)

    tt_cb_model = CatBoostRegressor(loss_function='Huber:delta=1.0', depth=6, verbose=1)
    tt_cb_model.fit(tt_X_train_processed, tt_y_train_scaled)
    tt_cb_model.save_model("models/tt_cat_boost_model.cbm")

    dump(tt_preprocessor, 'preprocessors/tt_data_preprocessor.joblib')
    dump(tt_y_scaler, 'preprocessors/tt_y_scaler.joblib')

    cb_y_pred = cb_model.predict(X_test_processed).reshape(-1, 1)

    rf_train_preds = cb_model.predict(X_train_processed).reshape(-1, 1)
    rf_test_preds = cb_model.predict(X_test_processed).reshape(-1, 1)
    rf_val_preds = cb_model.predict(X_val_processed).reshape(-1, 1)

    X_train_catnn = np.hstack([X_train_processed, rf_train_preds])
    X_val_catnn = np.hstack([X_val_processed, rf_val_preds])
    X_test_catnn = np.hstack([X_test_processed, rf_test_preds])

    def new_model_builder(hp):
        model = tf.keras.Sequential()

        input_dim = X_train_catnn.shape[1]

        # Гиперпараметры для нейронов и dropout
        hp_units1 = hp.Int('units1', 64, 1024, step=64)
        hp_units2 = hp.Int('units2', 16, 512, step=16)
        hp_dropout = hp.Float('dropout', 0.1, 0.5, step=0.1)

        # Новые гиперпараметры для регуляризации
        reg_type = hp.Choice('reg_type', ['l1', 'l2', 'l1l2'])
        reg_factor = hp.Float('reg_factor', 1e-5, 1e-3, sampling='log')
        lr = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
        activation = hp.Choice('activation', ['leaky_relu', 'relu', 'swish', 'tanh'])
        use_bn = hp.Boolean('batch_norm')

        if reg_type == 'l1':
            reg = tf.keras.regularizers.L1(reg_factor)
        elif reg_type == 'l2':
            reg = tf.keras.regularizers.L2(reg_factor)
        else:
            reg = tf.keras.regularizers.L1L2(l1=reg_factor, l2=reg_factor)

        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        model.add(tf.keras.layers.Dense(hp_units1, kernel_regularizer=reg))
        if activation == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU())
        else:
            model.add(tf.keras.layers.Activation(activation))

        if use_bn:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(hp_dropout))

        model.add(tf.keras.layers.Dense(hp_units2, kernel_regularizer=reg))
        if activation == 'leaky_relu':
            model.add(tf.keras.layers.LeakyReLU())
        else:
            model.add(tf.keras.layers.Activation(activation))

        model.add(tf.keras.layers.Dense(1, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
            loss=huber_loss,
            metrics=['mse', 'mae', R2Score()]
        )

        return model

    # настройка поиска гиперпараметров
    tuner = kt.BayesianOptimization(
        new_model_builder,
        objective='val_loss',
        max_trials=50,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='f1_laptime'
    )

    def get_callbacks():
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        return [early_stop, reduce_lr, checkpoint]

    tuner.search(
        X_train_catnn, y_train_scaled,
        epochs=300,
        validation_data=(X_val_catnn, y_val_scaled),
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0]

    val_loss = best_model.evaluate(X_val_catnn, y_val_scaled, verbose=1)
    print(f"Validation loss: {val_loss}")

    best_model.summary()
    print(best_hps.values)

    new_model = tuner.hypermodel.build(best_hps)
    new_model.set_weights(best_model.get_weights())

    history = new_model.fit(
        X_train_catnn, y_train_scaled,
        epochs=200,
        validation_data=(X_val_catnn, y_val_scaled),
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1
    )

    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = range(1, len(loss_values) + 1)

    best_model.save('models/best_laptime_model.keras')
    new_model.save('models/new_laptime_model.keras')

    val_loss_values = history_dict['val_loss']
    val_epochs = range(1, len(val_loss_values) + 1)

    # сравнение потерь на тренировочном и тестовом наборе данных при дообучении
    plt.plot(val_epochs, val_loss_values, 'r', label='Val loss')
    plt.plot(epochs, loss_values, 'b', label='Loss')
    plt.title('Сравнение потерь', fontsize=12)
    plt.xlabel('Эпохи', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

    loaded_model = tf.keras.models.load_model("models/new_laptime_model.keras")
    print(f"Neuro model")

    test_loss, test_mae, test_mse, test_r_squred = loaded_model.evaluate(X_test_catnn, y_test_scaled)
    print(f"Test Huber: {test_loss}, Test MAE: {test_mae}, Test R squared: {test_r_squred}, Test MSE:"
          f" {test_mse}")

    y_pred_neuro = loaded_model.predict(X_test_catnn)

    r2 = r2score(y_test_scaled, y_pred_neuro)
    print(f"R² Score: {r2}")

    y_test_inversed_neuro = y_scaler.inverse_transform(y_test_scaled)
    y_pred_inversed_neuro = y_scaler.inverse_transform(y_pred_neuro)

    mse_error = mean_squared_error(y_test_inversed_neuro, y_pred_inversed_neuro)
    print(f"MSE error: {mse_error}")

    huber = huber_loss(y_test_inversed_neuro, y_pred_inversed_neuro)
    print(f"Huber loss: {huber}")

    mae_error = mean_absolute_error(y_test_inversed_neuro, y_pred_inversed_neuro)
    print(f"MAE error: {mae_error}")

    mape_error = mean_absolute_percentage_error(y_test_inversed_neuro, y_pred_inversed_neuro)
    print(f"MAPE loss: {mape_error}")

    print(f"CatBoostRegressor")

    y_test_inversed_cat = y_scaler.inverse_transform(y_test_scaled)
    y_pred_inversed_cat = y_scaler.inverse_transform(cb_y_pred.reshape(-1, 1))

    r2 = r2score(y_test_inversed_cat, y_pred_inversed_cat)
    print(f"R² Score: {r2}")

    mse_error = mean_squared_error(y_test_inversed_cat, y_pred_inversed_cat)
    print(f"MSE error: {mse_error}")

    huber = huber_loss(y_test_inversed_cat, y_pred_inversed_cat)
    print(f"Huber loss: {huber}")

    mae_error = mean_absolute_error(y_test_inversed_cat, y_pred_inversed_cat)
    print(f"MAE error: {mae_error}")

    mape_error = mean_absolute_percentage_error(y_test_inversed_cat, y_pred_inversed_cat)
    print(f"MAPE loss: {mape_error}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_inversed_cat, y_pred_inversed_neuro, alpha=0.5)
    plt.plot([y_test_inversed_cat.min(), y_pred_inversed_neuro.max()],
             [y_test_inversed_cat.min(), y_pred_inversed_neuro.max()], 'k--', lw=2)
    plt.xlabel("Реальное время круга (сек)")
    plt.ylabel("Спрогнозированное время круга (сек)")
    plt.title("Спрогнозированные vs Реальные значения времен")
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    axes[0].scatter(y_test_inversed_cat, y_pred_inversed_neuro, alpha=0.5, color='blue', label='Нейросеть')
    axes[0].plot([y_test_inversed_cat.min(), y_pred_inversed_neuro.max()],
                 [y_test_inversed_cat.min(), y_pred_inversed_neuro.max()],
                 'k--', lw=2)
    axes[0].set_xlabel("Реальное время круга (сек)")
    axes[0].set_ylabel("Предсказанное время круга (сек)")
    axes[0].set_title("Нейросеть: Предсказания vs Реальные значения")
    axes[0].grid(True)

    axes[1].scatter(y_test_inversed_cat, y_pred_inversed_cat, alpha=0.5, color='red', label='CatBoostRegressor')
    axes[1].plot([y_test_inversed_cat.min(), y_pred_inversed_cat.max()],
                 [y_test_inversed_cat.min(), y_pred_inversed_cat.max()],
                 'k--', lw=2)
    axes[1].set_xlabel("Реальное время круга (сек)")
    axes[1].set_title("CatBoostRegressor: Предсказания vs Реальные значения")
    axes[1].grid(True)

    plt.suptitle("Сравнение моделей", fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()

    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

    tf.keras.utils.plot_model(loaded_model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

    explainer = shap.DeepExplainer(loaded_model, X_train_catnn)
    shap_values = explainer.shap_values(X_train_catnn)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    cb_feature_names.append("CB_predict")
    print(cb_feature_names)
    feature_importance = pd.DataFrame({
        "Feature": cb_feature_names,
        "Mean|SHAP|": mean_abs_shap.flatten()
    }).sort_values(by="Mean|SHAP|", ascending=True)

    top_unuseful = feature_importance.head(100)
    print("Самые бесполезные признаки:\n", top_unuseful)

    # Визуализация
    plt.figure(figsize=(20, 6))
    plt.barh(top_unuseful["Feature"], top_unuseful["Mean|SHAP|"])
    plt.xlabel("Средний абсолютный SHAP")
    plt.title("Топ бесполезных признаков")
    plt.show()


def load_single_file(file_path: str) -> pd.DataFrame:
    try:
        encodings = ['utf-8', 'latin1', 'windows-1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=',',
                    encoding=encoding,
                    engine='python',
                    skipinitialspace=True,
                    quotechar='"',
                    on_bad_lines='warn'
                )
                if not df.empty:
                    return df
            except Exception as e:
                continue
        raise ValueError(f"Не удалось прочитать файл {file_path}")
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки {file_path}: {str(e)}")


if __name__ == '__main__':
    main()
