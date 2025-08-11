import pandas as pd
from main import load_single_file, huber_loss
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

import tensorflow as tf

matplotlib.use('TkAgg')


def analyse_main():
    df = load_single_file("C:/Users/PC/Desktop/f1_py/dataset/QualiData.csv")

    # delete_anomalies(df)
    # correlation_heatmap(df)

    delete_anomalies(df)
    #correlation_heatmap(df)
    # show_central_tendency(df)
    # show_variability_metrics(df)
    # bootstrapAnalyze(df)
    # correlationAnalyze(df)
    #linear_regression(df)

def correlation_heatmap(df):
    df = df.dropna(subset=['LapTime'])

    df["IsStreetCircuit"] = df["IsStreetCircuit"].astype(int)
    df["Rainfall"] = df["Rainfall"].astype(int)
    df["IsHighHumidity"] = df["IsHighHumidity"].astype(int)

    # Оставим только числовые признаки (не категориальные)
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Построим матрицу корреляций
    corr_matrix = numeric_df.corr(method='pearson')

    # Настроим размеры графика
    plt.figure(figsize=(16, 12))

    # Отрисуем тепловую карту
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)

    # Подписи и заголовок
    plt.title("Тепловая карта корреляций признаков", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Сохранение в файл (если вставлять в ВКР)
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.show()


def delete_anomalies(df):
    # Определим признаки
    target_col = 'LapTime'

    # Инициализируем Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    # Обучаем модель на целевой переменной
    preds_1d = iso.fit_predict(df[[target_col]])
    # Оставляем только те строки, которые не было помечены как выбросы
    df_laptime = df[preds_1d != -1]

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(x=df['LapTime'], ax=axes[0])
    axes[0].set_title("До Isolation Forest")

    sns.boxplot(x=df_laptime[target_col], ax=axes[1])
    axes[1].set_title(
        f"После Isolation Forest\nУдалено: {len(df) - len(df_laptime)} строк")

    # Сохранение результата
    df_laptime.to_csv("dataset/QualiData_cleaned.csv", index=False)

    plt.tight_layout()
    plt.show()

    # Статистика
    print(f"До фильтрации: {len(df)} строк")
    print(f"После фильтрации: {len(df_laptime)} строк")

def analyse_low(df):
    # проверка пропущенных значений
    missing_values = df.isnull().sum()
    print("Пропущенные значения в каждом столбце:")
    print(missing_values)

    # Предположим, df — это исходный DataFrame
    numerical_features = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'DriverPoints', 'TeamPoints', 'CircuitLength']

    # До нормализации
    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df[numerical_features])
    plt.title("До масштабирования")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # После нормализации
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)

    plt.figure(figsize=(14, 5))
    sns.boxplot(data=df_scaled)
    plt.title("После масштабирования (RobustScaler)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def bootstrapAnalyze(df):
    # Бутстрап тут
    bootstrap_means = bootstrap(
        df['LapTime'],
        statistic=np.mean,
        n_iterations=1000000,
        random_seed=42
    )

    # Вычисление краев доверительного интервала
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    mean_value = np.mean(bootstrap_means)
    std_dev = np.std(bootstrap_means)

    plt.figure(figsize=(12, 7))
    sns.histplot(bootstrap_means, kde=True, color="blue", label="Распределение средних")

    # Добавление линий и аннотаций
    plt.axvline(mean_value, color='yellow', linestyle='-', linewidth=2,
                label=f'Среднее: {mean_value:.3f}')
    plt.axvline(lower_bound, color='crimson', linestyle='--', linewidth=2,
                label=f'Нижняя граница ДИ: {lower_bound:.3f}')
    plt.axvline(upper_bound, color='forestgreen', linestyle='--', linewidth=2,
                label=f'Верхняя граница ДИ: {upper_bound:.3f}')

    plt.title('Распределение выборочных средних', pad=20)
    plt.xlabel('Среднее значение', labelpad=10)
    plt.ylabel('Частота', labelpad=10)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print(f"• Среднее значение: {mean_value:.3f} ± {std_dev:.3f}")
    print(f"• 95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"• Размах доверительного интервала: {upper_bound - lower_bound:.3f}")


def bootstrap(data, statistic, n_iterations=1000, sample_size=None, random_seed=None):
    if len(data) == 0:
        raise ValueError("Данные не могут быть пустыми")

    if sample_size is not None and sample_size <= 0:
        raise ValueError("Размер выборки должен быть положительным числом")

    if random_seed is not None:
        np.random.seed(random_seed)

    sample_size = sample_size or len(data)

    bootstrap_samples = np.zeros(n_iterations)

    for i in tqdm(range(n_iterations), desc="Бутстрап-итерации"):
        sample = np.random.choice(data, size=sample_size, replace=True)
        bootstrap_samples[i] = statistic(sample)

    return bootstrap_samples


def correlationAnalyze(df):
    features = ['LapTime', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'Team', 'Driver', 'DriverAvgQualiPos',
                'TeamAvgQualiPos', 'DriverPoints', 'TeamPoints', 'Rainfall', 'Year', 'F1Era',
                'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons', 'TeamPointsContribution']

    X = df[features]

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

    # Визуализация корреляционных матриц
    plt.figure(figsize=(22, 10))
    sns.heatmap(pearson_corr,
                annot=True,
                xticklabels=True,
                yticklabels=True,
                cmap='coolwarm',
                fmt='.2f',
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor='white')
    plt.title(f'Корреляция Пирсона', pad=20)

    plt.figure(figsize=(22, 10))
    sns.heatmap(spearman_corr,
                annot=True,
                xticklabels=True,
                yticklabels=True,
                cmap='coolwarm',
                fmt='.2f',
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor='white')
    plt.title(f'Корреляция Спирмена', pad=20)
    plt.show()

    # Поиск связей с числом продаж во всем мире
    print("\nКорреляции с временем круга в сек.:")
    time_corr = spearman_corr['LapTime'].sort_values(key=abs, ascending=False)
    print(time_corr[1:(time_corr.size - 1)])


def linear_regression(df):
    features = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'Team', 'Driver', 'DriverAvgQualiPos',
                'TeamAvgQualiPos', 'DriverPoints', 'TeamPoints', 'Rainfall', 'Year', 'F1Era',
                'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons', 'TeamPointsContribution']

    X = df[features]

    categorical_features = ['Team', 'Driver', 'Rainfall', 'F1Era', 'IsHighHumidity']
    numeric_features = ['LapTime', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'DriverAvgQualiPos',
                        'TeamAvgQualiPos',
                        'DriverPoints', 'TeamPoints', 'Year', 'TrackAirDiff', 'DriverQualiPace', 'DriverSeasons',
                        'TeamPointsContribution']

    dum_driver = pd.get_dummies(X['Driver'], prefix="Driver", dtype=int)
    dum_team = pd.get_dummies(X['Team'], prefix="Team", dtype=int)
    dum_rainfall = pd.get_dummies(X['Rainfall'], prefix="Rainfall", dtype=int)
    dum_f1era = pd.get_dummies(X['F1Era'], prefix="F1Era", dtype=int)
    dum_highhumidity = pd.get_dummies(X['IsHighHumidity'], prefix="IsHighHumidity", dtype=int)

    x_value = pd.concat([X, dum_driver, dum_team, dum_rainfall, dum_f1era, dum_highhumidity], axis=1) \
        .drop(columns=['Driver', 'Team', 'Rainfall', 'F1Era', 'IsHighHumidity'])

    x = x_value
    y = df['LapTime']

    # Добавляем константу (intercept)
    x = sm.add_constant(x, has_constant='add')

    # Строим и обучаем модель
    model = sm.OLS(y.astype(float), x.astype(float)).fit()

    # Выводим полную статистику модели
    print("\nРезультаты регрессионного анализа:")
    print(model.summary())

    # predictions = model.predict(new_df[:])

    # Визуализация коэффициентов
    plt.figure(figsize=(22, 10))
    coefs = model.params[1:]  # Исключаем intercept

    # Создаем DataFrame для визуализации
    coef_df = pd.DataFrame({
        'Feature': x_value.columns.tolist(),
        'Coefficient': coefs,
    }).sort_values('Coefficient', key=abs, ascending=False)

    # График коэффициентов с доверительными интервалами
    plt.errorbar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'],
        fmt='o',
        color='b',
        ecolor='r',
        capsize=5
    )
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title(f'Коэффициенты линейной регрессии')
    plt.xlabel('Значение коэффициента')
    plt.ylabel('Признак')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    df['pred'] = model.predict()
    huber = huber_loss(df['LapTime'], df['pred'])
    print(f"Huber loss: {tf.reduce_mean(huber)}")


def show_central_tendency(df: pd.DataFrame) -> None:
    numerical_columns = ['LapTime', 'DriverAvgQualiPos', 'TeamAvgQualiPos']

    existing_columns = [col for col in numerical_columns if col in df.columns]
    central_tendency = pd.DataFrame(
        index=existing_columns,
        columns=['Mean', 'Median', 'Mode']
    )

    for col in existing_columns:
        clean_series = df[col].dropna()
        if clean_series.empty:
            continue

        mean_value = clean_series.mean()
        median_value = clean_series.median()
        try:
            mode_value = stats.mode(clean_series, keepdims=True).mode[0]
        except:
            mode_value = 'N/A'

        central_tendency.loc[col] = [mean_value, median_value, mode_value]

    print("\n" + "=" * 60)
    print(f"Оценка центрального положения данных")
    print("=" * 60)
    print(central_tendency.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 60 + "\n")


def show_variability_metrics(df: pd.DataFrame) -> None:
    numerical_columns = ['LapTime', 'DriverAvgQualiPos', 'TeamAvgQualiPos']

    existing_columns = [col for col in numerical_columns if col in df.columns]

    variability_metrics = pd.DataFrame(
        index=existing_columns,
        columns=['Range', 'Variance', 'Standard Deviation']
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
    print(f"Метрики вариабельности данных")
    print("=" * 65)
    print(variability_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    analyse_main()
