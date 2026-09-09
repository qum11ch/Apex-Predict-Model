from typing import Callable, Optional

import pandas as pd
from pathlib import Path

from main import load_single_file, huber_loss
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm

import tensorflow as tf

matplotlib.use('TkAgg')
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"


def analyse_main():
    current_year = 2026
    df = load_single_file(DATASET_DIR / f"QualiData_{current_year}.csv")

    df_cleaned = load_single_file(DATASET_DIR / f"QualiData_{current_year}_cleaned.csv")
    # df_cleaned = delete_anomalies(df, current_year)

    correlation_heatmap(df_cleaned)
    # show_central_tendency(df_cleaned)
    # show_variability_metrics(df_cleaned)
    # bootstrap_analyze(df_cleaned["LapTime"], n_iterations=1000)
    # correlation_analyze(df_cleaned)
    # linear_regression(df_cleaned)


def delete_anomalies(
        df: pd.DataFrame,
        currentYear: int,
        target_col: str = "LapTime",
        contamination: float = 0.01,
        random_state: int = 42, ) -> pd.DataFrame:
    """
    Удаляет выбросы в целевой переменной с помощью Isolation Forest.

    Returns
    -------
    pd.DataFrame
        Очищенный DataFrame.
    """
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state)

    preds_1d = iso.fit_predict(df[[target_col]])
    df_cleaned = df[preds_1d != -1].copy()

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(x=df[target_col], ax=axes[0])
    axes[0].set_title("До Isolation Forest")

    sns.boxplot(x=df_cleaned[target_col], ax=axes[1])
    axes[1].set_title(
        f"После Isolation Forest\nУдалено: {len(df) - len(df_cleaned)} строк"
    )

    plt.tight_layout()
    plt.show()

    print(f"До фильтрации: {len(df)} строк")
    print(f"После фильтрации: {len(df_cleaned)} строк")

    out_path = DATASET_DIR / f"QualiData_{currentYear}_cleaned.csv"
    df_cleaned.to_csv(out_path, index=False)

    return df_cleaned


# Описательная статистика
def show_central_tendency(df: pd.DataFrame) -> None:
    numerical_columns = ["LapTime", "DriverAvgQualiPos", "TeamAvgQualiPos"]
    existing_columns = [col for col in numerical_columns if col in df.columns]

    central_tendency = pd.DataFrame(
        index=existing_columns,
        columns=["Mean", "Median", "Mode"],
    )

    for col in existing_columns:
        clean_series = df[col].dropna()
        if clean_series.empty:
            continue

        mean_value = clean_series.mean()
        median_value = clean_series.median()

        try:
            mode_result = stats.mode(clean_series, keepdims=True)
            mode_value = mode_result.mode[0]
        except Exception:
            mode_value = "N/A"

        central_tendency.loc[col] = [mean_value, median_value, mode_value]

    print("\n" + "=" * 60)
    print("Оценка центрального положения данных")
    print("=" * 60)
    print(central_tendency.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 60 + "\n")


def show_variability_metrics(df: pd.DataFrame) -> None:
    numerical_columns = ["LapTime", "DriverAvgQualiPos", "TeamAvgQualiPos"]
    existing_columns = [col for col in numerical_columns if col in df.columns]

    variability_metrics = pd.DataFrame(
        index=existing_columns,
        columns=["Range", "Variance", "Standard Deviation"],
    )

    for col in existing_columns:
        clean_series = df[col].dropna()
        if clean_series.empty:
            continue

        range_value = clean_series.max() - clean_series.min()
        variance_value = clean_series.var()
        std_dev_value = clean_series.std()

        variability_metrics.loc[col] = [range_value, variance_value, std_dev_value]

    print("\n" + "=" * 65)
    print("Метрики вариабельности данных")
    print("=" * 65)
    print(variability_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 65 + "\n")


# Визуализация корреляций
def correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    df = df.dropna(subset=["LapTime"])

    for col in ["IsStreetCircuit", "Rainfall", "IsHighHumidity"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )

    plt.title("Тепловая карта корреляций признаков", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# Бутстрап
def bootstrap(
        data: np.ndarray | pd.Series,
        statistic: Callable[[np.ndarray], float],
        n_iterations: int = 1000,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None, ) -> np.ndarray:
    """
        Выполняет бутстрап-оценку заданной статистики.

        Parameters
        ----------
        data : np.ndarray | pd.Series
            Исходные данные (1D).
        statistic : callable
            Функция, принимающая массив и возвращающая скаляр (например, np.mean).
        n_iterations : int
            Количество бутстрап-выборок.
        sample_size : int, optional
            Размер каждой выборки. По умолчанию равен len(data).
        random_seed : int, optional
            Seed для воспроизводимости.

        Returns
        -------
        np.ndarray
            Массив значений статистики для каждой бутстрап-выборки.
    """
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("Данные должны быть одномерными")

    if len(data) == 0:
        raise ValueError("Данные не могут быть пустыми")

    if n_iterations <= 0:
        raise ValueError("n_iterations должен быть положительным числом")

    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size должен быть положительным числом")

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    sample_size = sample_size or len(data)

    bootstrap_samples = np.empty(n_iterations, dtype=float)

    for i in tqdm(range(n_iterations), desc="Бутстрап-итерации"):
        sample = rng.choice(data, size=sample_size, replace=True)
        bootstrap_samples[i] = statistic(sample)

    return bootstrap_samples


def bootstrap_analyze(
        data: np.ndarray | pd.Series,
        statistic: Callable[[np.ndarray], float] = np.mean,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None, ) -> dict:
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Данные не могут быть пустыми")

    bootstrap_means = bootstrap(
        data,
        statistic=statistic,
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    mean_value = float(np.mean(bootstrap_means))
    std_dev = float(np.std(bootstrap_means, ddof=0))

    plt.figure(figsize=(12, 7))
    sns.histplot(bootstrap_means, kde=True, color="blue", label="Распределение средних")

    plt.axvline(
        mean_value,
        color="yellow",
        linestyle="-",
        linewidth=2,
        label=f"Среднее: {mean_value:.3f}",
    )
    plt.axvline(
        lower_bound,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Нижняя граница ДИ: {lower_bound:.3f}",
    )
    plt.axvline(
        upper_bound,
        color="forestgreen",
        linestyle="--",
        linewidth=2,
        label=f"Верхняя граница ДИ: {upper_bound:.3f}",
    )

    plt.title("Распределение выборочных средних", pad=20)
    plt.xlabel("Среднее значение", labelpad=10)
    plt.ylabel("Частота", labelpad=10)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print(f"Среднее значение: {mean_value:.3f} ± {std_dev:.3f}")
    print(
        f"{int(confidence_level * 100)}% доверительный интервал: "
        f"[{lower_bound:.3f}, {upper_bound:.3f}]"
    )
    print(f"Размах доверительного интервала: {upper_bound - lower_bound:.3f}")

    return {
        "mean": mean_value,
        "std": std_dev,
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
        "bootstrap_samples": bootstrap_means,
    }


# Корреляционный анализ
def correlationAnalyze(df: pd.DataFrame) -> None:
    features = ['LapTime', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'Team', 'Driver', 'DriverAvgQualiPos',
                'TeamAvgQualiPos', 'DriverPoints', 'TeamPoints', 'Rainfall', 'Year', 'F1Era',
                'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons', 'TeamPointsContribution']

    X = df[features].copy()

    dum_driver = pd.get_dummies(X['Driver'], prefix="Driver", dtype=int)
    dum_team = pd.get_dummies(X['Team'], prefix="Team", dtype=int)
    dum_rainfall = pd.get_dummies(X['Rainfall'], prefix="Rainfall", dtype=int)
    dum_f1era = pd.get_dummies(X['F1Era'], prefix="F1Era", dtype=int)
    dum_highhumidity = pd.get_dummies(X['IsHighHumidity'], prefix="IsHighHumidity", dtype=int)

    x_value = pd.concat([X, dum_driver, dum_team, dum_rainfall, dum_f1era, dum_highhumidity], axis=1) \
        .drop(columns=['Driver', 'Team', 'Rainfall', 'F1Era', 'IsHighHumidity'])

    pearson_corr = x_value.corr(method='pearson', numeric_only=True)
    spearman_corr = x_value.corr(method='spearman', numeric_only=True)

    print("\nКорреляция Пирсона:")
    print(pearson_corr)

    print("\nКорреляция Спирмена:")
    print(spearman_corr)

    plt.figure(figsize=(22, 10))
    sns.heatmap(
        pearson_corr,
        annot=True,
        xticklabels=True,
        yticklabels=True,
        cmap='coolwarm',
        fmt='.2f',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f'Корреляция Пирсона', pad=20)

    plt.figure(figsize=(22, 10))
    sns.heatmap(
        spearman_corr,
        annot=True,
        xticklabels=True,
        yticklabels=True,
        cmap='coolwarm',
        fmt='.2f',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f'Корреляция Спирмена', pad=20)

    plt.show()

    print("\nКорреляции с временем круга в сек.:")
    time_corr = spearman_corr['LapTime'].sort_values(key=abs, ascending=False)
    print(time_corr[1:(time_corr.size - 1)])


# Линейная регрессия
def linear_regression(df: pd.DataFrame) -> None:
    features = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'Team', 'Driver', 'DriverAvgQualiPos',
                'TeamAvgQualiPos', 'DriverPoints', 'TeamPoints', 'Rainfall', 'Year', 'F1Era',
                'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons', 'TeamPointsContribution']

    X = df[features].copy()

    dum_driver = pd.get_dummies(X["Driver"], prefix="Driver", dtype=int)
    dum_team = pd.get_dummies(X["Team"], prefix="Team", dtype=int)
    dum_rainfall = pd.get_dummies(X["Rainfall"], prefix="Rainfall", dtype=int)
    dum_f1era = pd.get_dummies(X["F1Era"], prefix="F1Era", dtype=int)
    dum_highhumidity = pd.get_dummies(
        X["IsHighHumidity"], prefix="IsHighHumidity", dtype=int
    )

    x_value = pd.concat(
        [X, dum_driver, dum_team, dum_rainfall, dum_f1era, dum_highhumidity],
        axis=1,
    ).drop(columns=["Driver", "Team", "Rainfall", "F1Era", "IsHighHumidity"])

    x = sm.add_constant(x_value, has_constant="add")
    y = df["LapTime"].astype(float)

    model = sm.OLS(y, x.astype(float)).fit()

    print("\nРезультаты регрессионного анализа:")
    print(model.summary())

    # Визуализация коэффициентов
    plt.figure(figsize=(22, 10))
    coefs = model.params[1:]  # без intercept

    coef_df = pd.DataFrame(
        {
            "Feature": x_value.columns.tolist(),
            "Coefficient": coefs,
        }
    ).sort_values("Coefficient", key=abs, ascending=False)

    plt.errorbar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        fmt="o",
        color="b",
        ecolor="r",
        capsize=5,
    )
    plt.axvline(x=0, color="gray", linestyle="--")
    plt.title("Коэффициенты линейной регрессии")
    plt.xlabel("Значение коэффициента")
    plt.ylabel("Признак")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    df = df.copy()
    df["pred"] = model.predict()

    if "huber_loss" in globals():
        huber = huber_loss(df["LapTime"], df["pred"])
        # Если huber_loss возвращает массив, берём среднее
        huber_mean = float(tf.reduce_mean(huber).numpy())
        print(f"Huber loss: {huber_mean:.6f}")
    else:
        print("Функция huber_loss не найдена, пропускаем расчёт Huber loss.")


# Анализ низких значений
def analyse_low(df: pd.DataFrame) -> None:
    missing_values = df.isnull().sum()
    print("Пропущенные значения в каждом столбце:")
    print(missing_values)

    numerical_features = [
        "AirTemp",
        "Humidity",
        "Pressure",
        "TrackTemp",
        "DriverPoints",
        "TeamPoints",
        "CircuitLength",
    ]

    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df[numerical_features])
    plt.title("До масштабирования")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    scaler = RobustScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]),
        columns=numerical_features,
    )

    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df_scaled)
    plt.title("После масштабирования (RobustScaler)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyse_main()
