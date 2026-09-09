from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from fastf1.ergast import Ergast

from catboost import CatBoostRegressor

from dataset import (
    f1_era,
    prev_team_id,
    get_driver_seasons,
    get_engine_seasons,
    get_driver_standings,
    get_constructor_standings,
    get_current_season_quali,
    get_circuits_len,
    add_season_stats
)
from main import R2Score, r2score, huber_loss

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
PREPROC_DIR = ROOT / "preprocessors"
MODEL_DIR = ROOT / "models"

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def time_to_seconds(seconds: float) -> str:
    if pd.isna(seconds):
        return np.nan
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes}:{remaining_seconds:06.3f}"


class ModelsCache:
    def __init__(self) -> None:
        self.loaded_model = None
        self.cb_model = None
        self.tt_cb_model = None
        self.pres_cb_model = None
        self.at_cb_model = None
        self.preprocessor = None
        self.y_scaler = None
        self.tt_preprocessor = None
        self.tt_y_scaler = None
        self.pres_preprocessor = None
        self.pres_y_scaler = None
        self.at_preprocessor = None
        self.at_y_scaler = None
        self._loaded = False

    def load_all(self) -> None:
        if self._loaded:
            return

        self.loaded_model = tf.keras.models.load_model(
            MODEL_DIR / "best_laptime_model.keras",
            custom_objects={"huber_loss": huber_loss, "R2Score": R2Score},
            compile=False,
        )

        self.cb_model = CatBoostRegressor().load_model(MODEL_DIR / "cat_boost_model.cbm")
        self.tt_cb_model = CatBoostRegressor().load_model(MODEL_DIR / "tt_cat_boost_model.cbm")
        self.pres_cb_model = CatBoostRegressor().load_model(MODEL_DIR / "pres_cat_boost_model.cbm")
        self.at_cb_model = CatBoostRegressor().load_model(MODEL_DIR / "at_cat_boost_model.cbm")

        self.preprocessor = load(PREPROC_DIR / "data_preprocessor.joblib")
        self.y_scaler = load(PREPROC_DIR / "y_scaler.joblib")
        self.tt_preprocessor = load(PREPROC_DIR / "tt_data_preprocessor.joblib")
        self.tt_y_scaler = load(PREPROC_DIR / "tt_y_scaler.joblib")
        self.pres_preprocessor = load(PREPROC_DIR / "pres_data_preprocessor.joblib")
        self.pres_y_scaler = load(PREPROC_DIR / "pres_y_scaler.joblib")
        self.at_preprocessor = load(PREPROC_DIR / "at_data_preprocessor.joblib")
        self.at_y_scaler = load(PREPROC_DIR / "at_y_scaler.joblib")

        self._loaded = True


models_cache = ModelsCache()


def get_prediction(
        year: int,
        air_temp: float,
        pressure: float,
        humidity: float,
        rainfall: int,
        event: str,
        gp_round: int,
        df: pd.DataFrame,
        circuit_corners: int,
        circuit_length: float,
        is_street_circuit: bool,
        circuit_id: str, ) -> pd.DataFrame:
    """
    Прогнозирует результаты квалификации (Q1, Q2, Q3).
    """
    ergast = Ergast(result_type="pandas", auto_cast=True)
    models_cache.load_all()

    current_season_quali = None
    if year > 2024:
        current_season_quali = get_current_season_quali(ergast, year, gp_round)

    driver = get_driver_standings(ergast, year, gp_round)
    constructor = get_constructor_standings(ergast, year, gp_round)

    df = add_season_stats(
        df, driver, constructor, gp_round, year, current_season_quali, event, ergast
    )

    first_event_name = event + "1"

    df["Year"] = year
    df["IsStreetCircuit"] = is_street_circuit
    df["F1Era"] = f1_era(year)
    df["CircuitCorners"] = circuit_corners
    df["CircuitLength"] = circuit_length
    df["Event"] = first_event_name
    df["CircuitId"] = circuit_id

    # Прогнозирование TrackTemp для Q1
    is_high_humidity = 1 if humidity > 70.0 else 0

    tt_data_event1 = {
        "Year": [year],
        "AirTemp": [air_temp],
        "Humidity": [humidity],
        "Pressure": [pressure],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [first_event_name],
    }

    tt_df_event1 = pd.DataFrame(tt_data_event1)
    tt_data_event1 = models_cache.tt_preprocessor.transform(tt_df_event1)

    track_temp_normal_event1 = models_cache.tt_cb_model.predict(tt_data_event1)
    track_temp_event1 = models_cache.tt_y_scaler.inverse_transform(
        track_temp_normal_event1.reshape(-1, 1)
    ).item()
    track_temp_event1 = round(track_temp_event1, 1)

    track_air_diff = round(track_temp_event1 - air_temp, 1)

    df["AirTemp"] = air_temp
    df["Humidity"] = humidity
    df["Pressure"] = pressure
    df["Rainfall"] = rainfall
    df["TrackTemp"] = track_temp_event1
    df["TrackAirDiff"] = track_air_diff
    df["IsHighHumidity"] = is_high_humidity

    df["DriverQualiPace"] = None
    df["DriverSeasons"] = None
    df["TeamPointsContribution"] = None
    df["CarEngine"] = None

    drivers_season_df = get_driver_seasons(season=year)
    engine_season_df = get_engine_seasons(season=year)

    for i, row in df.iterrows():
        driver_avg_quali = row["DriverAvgQualiPos"]
        team_avg_quali = row["TeamAvgQualiPos"]
        driver_code = row["Driver"]
        driver_points = row["DriverPoints"]
        team_points = row["TeamPoints"]
        team_name = row["Team"]

        if driver_avg_quali != 0.0:
            driver_quali_pace = round((team_avg_quali - driver_avg_quali), 2)
        else:
            driver_quali_pace = 0.0

        if team_points != 0.0:
            team_points_contribution = round((driver_points / team_points), 2)
        else:
            team_points_contribution = 0.5

        df.at[i, "TeamPointsContribution"] = team_points_contribution
        df.at[i, "DriverQualiPace"] = driver_quali_pace

        seasons_count = drivers_season_df[
            drivers_season_df["driverCode"] == driver_code
            ]["seasonsCount"].values

        car_engine = engine_season_df[
            engine_season_df["teamName"] == team_name
            ]["engine"].values

        if car_engine.size != 0:
            df.at[i, "CarEngine"] = car_engine.item()
        else:
            df.at[i, "CarEngine"] = None

        if seasons_count.size != 0:
            df.at[i, "DriverSeasons"] = seasons_count.item()
        else:
            k = year - 1
            while k > 2017:
                prev_drivers_season_df = get_driver_seasons(season=k)
                new_seasons_count = prev_drivers_season_df[
                    prev_drivers_season_df["driverCode"] == driver_code
                    ]["seasonsCount"].values
                if new_seasons_count.size != 0:
                    df.at[i, "DriverSeasons"] = new_seasons_count.item()
                    break
                k -= 1
                if k == 2017:
                    df.at[i, "DriverSeasons"] = 0.0

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

    # Q1 Прогноз
    Q1_X = df[features]
    Q1_processed = models_cache.preprocessor.transform(Q1_X)
    cb_Q1_pred = models_cache.cb_model.predict(Q1_processed).reshape(-1, 1)
    X_Q1_catnn = np.hstack([Q1_processed, cb_Q1_pred])

    Q1_pred_neuro = models_cache.loaded_model.predict(X_Q1_catnn)
    Q1_pred_neuro_inversed = models_cache.y_scaler.inverse_transform(Q1_pred_neuro)

    df[first_event_name] = Q1_pred_neuro_inversed

    # Q2 Прогноз
    if year == 2026:
        Q2_df = df.sort_values(by=first_event_name).head(16)
    else:
        Q2_df = df.sort_values(by=first_event_name).head(15)

    second_event_name = event + "2"
    Q2_df["Event"] = second_event_name

    pres_data_event2 = {
        "Year": [year],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "AirTemp": [air_temp],
        "TrackTemp": [track_temp_event1],
        "CircuitId": [circuit_id],
        "Event": [second_event_name],
    }

    pres_df_event2 = pd.DataFrame(pres_data_event2)
    pres_data_event2 = models_cache.pres_preprocessor.transform(pres_df_event2)

    pressure_normal_event2 = models_cache.pres_cb_model.predict(pres_data_event2)
    pressure_event2 = models_cache.pres_y_scaler.inverse_transform(
        pressure_normal_event2.reshape(-1, 1)
    ).item()
    pressure_event2 = round(pressure_event2, 1)

    at_data_event2 = {
        "Year": [year],
        "Humidity": [humidity],
        "Pressure": [pressure_event2],
        "TrackTemp": [track_temp_event1],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [second_event_name],
    }

    at_df_event2 = pd.DataFrame(at_data_event2)
    at_data_event2 = models_cache.at_preprocessor.transform(at_df_event2)

    air_temp_normal_event2 = models_cache.at_cb_model.predict(at_data_event2)
    air_temp_event2 = models_cache.at_y_scaler.inverse_transform(
        air_temp_normal_event2.reshape(-1, 1)
    ).item()
    air_temp_event2 = round(air_temp_event2, 1)

    tt_data_event2 = {
        "Year": [year],
        "AirTemp": [air_temp_event2],
        "Humidity": [humidity],
        "Pressure": [pressure_event2],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [second_event_name],
    }

    tt_df_event2 = pd.DataFrame(tt_data_event2)
    tt_data_event2 = models_cache.tt_preprocessor.transform(tt_df_event2)

    track_temp_normal_event2 = models_cache.tt_cb_model.predict(tt_data_event2)
    track_temp_event2 = models_cache.tt_y_scaler.inverse_transform(
        track_temp_normal_event2.reshape(-1, 1)
    ).item()
    track_temp_event2 = round(track_temp_event2, 1)

    track_air_diff = round(track_temp_event2 - air_temp_event2, 1)

    Q2_df["AirTemp"] = air_temp_event2
    Q2_df["TrackTemp"] = track_temp_event2
    Q2_df["TrackAirDiff"] = track_air_diff
    Q2_df["Pressure"] = pressure_event2

    Q2_X = Q2_df[features]
    Q2_processed = models_cache.preprocessor.transform(Q2_X)
    cb_Q2_preds = models_cache.cb_model.predict(Q2_processed).reshape(-1, 1)
    X_Q2_catnn = np.hstack([Q2_processed, cb_Q2_preds])

    Q2_pred_neuro = models_cache.loaded_model.predict(X_Q2_catnn)
    Q2_pred_neuro_inversed = models_cache.y_scaler.inverse_transform(Q2_pred_neuro)

    Q2_df[second_event_name] = Q2_pred_neuro_inversed

    Q2_results = Q2_df.sort_values(by=second_event_name)[["Driver", second_event_name]]
    Q2_results = Q2_results.reset_index(drop=True)

    # Q3 Прогноз
    if year == 2026:
        Q3_df = Q2_df.sort_values(by=second_event_name).head(10)
    else:
        Q3_df = Q2_df.sort_values(by=second_event_name).head(10)

    third_event_name = event + "3"
    Q3_df["Event"] = third_event_name

    pres_data_event3 = {
        "Year": [year],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "AirTemp": [air_temp_event2],
        "TrackTemp": [track_temp_event2],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [third_event_name],
    }

    pres_df_event3 = pd.DataFrame(pres_data_event3)
    pres_data_event3 = models_cache.pres_preprocessor.transform(pres_df_event3)

    pressure_normal_event3 = models_cache.pres_cb_model.predict(pres_data_event3)
    pressure_event3 = models_cache.pres_y_scaler.inverse_transform(
        pressure_normal_event3.reshape(-1, 1)
    ).item()
    pressure_event3 = round(pressure_event3, 1)

    at_data_event3 = {
        "Year": [year],
        "TrackTemp": [track_temp_event2],
        "Humidity": [humidity],
        "Pressure": [pressure_event3],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [third_event_name],
    }

    at_df_event3 = pd.DataFrame(at_data_event3)
    at_data_event3 = models_cache.at_preprocessor.transform(at_df_event3)

    air_temp_normal_event3 = models_cache.at_cb_model.predict(at_data_event3)
    air_temp_event3 = models_cache.at_y_scaler.inverse_transform(
        air_temp_normal_event3.reshape(-1, 1)
    ).item()
    air_temp_event3 = round(air_temp_event3, 1)

    tt_data_event3 = {
        "Year": [year],
        "AirTemp": [air_temp_event3],
        "Humidity": [humidity],
        "Pressure": [pressure_event3],
        "Rainfall": [rainfall],
        "IsStreetCircuit": [is_street_circuit],
        "IsHighHumidity": [is_high_humidity],
        "CircuitId": [circuit_id],
        "Event": [third_event_name],
    }

    tt_df_event3 = pd.DataFrame(tt_data_event3)
    tt_data_event3 = models_cache.tt_preprocessor.transform(tt_df_event3)

    track_temp_normal_event3 = models_cache.tt_cb_model.predict(tt_data_event3)
    track_temp_event3 = models_cache.tt_y_scaler.inverse_transform(
        track_temp_normal_event3.reshape(-1, 1)
    ).item()
    track_temp_event3 = round(track_temp_event3, 1)

    track_air_diff = round(track_temp_event3 - air_temp_event3, 1)

    Q3_df["AirTemp"] = air_temp_event3
    Q3_df["TrackTemp"] = track_temp_event3
    Q3_df["TrackAirDiff"] = track_air_diff
    Q3_df["Pressure"] = pressure_event3

    Q3_x = Q3_df[features]
    Q3_processed = models_cache.preprocessor.transform(Q3_x)
    cb_Q3_preds = models_cache.cb_model.predict(Q3_processed).reshape(-1, 1)
    X_Q3_catnn = np.hstack([Q3_processed, cb_Q3_preds])

    Q3_pred_neuro = models_cache.loaded_model.predict(X_Q3_catnn)
    Q3_pred_neuro_inversed = models_cache.y_scaler.inverse_transform(Q3_pred_neuro)

    Q3_df[third_event_name] = Q3_pred_neuro_inversed

    Q3_result = Q3_df.sort_values(by=third_event_name)[["Driver", third_event_name]]
    Q3_result = Q3_result.reset_index(drop=True)

    # Результаты
    df[second_event_name] = None
    df[third_event_name] = None

    for i, row in df.iterrows():
        driver_code = row["Driver"]
        Q2_time = Q2_results[Q2_results["Driver"] == driver_code][second_event_name].values
        Q3_time = Q3_result[Q3_result["Driver"] == driver_code][third_event_name].values

        if len(Q2_time) == 0:
            Q2_time = np.nan
        if len(Q3_time) == 0:
            Q3_time = np.nan

        df.at[i, second_event_name] = Q2_time
        df.at[i, third_event_name] = Q3_time

    Q3_group = df[df[third_event_name].notna()].sort_values(by=third_event_name)
    Q2_group = df[
        (df[third_event_name].isna()) & (df[second_event_name].notna())
        ].sort_values(by=second_event_name)
    Q1_group = df[
        (df[third_event_name].isna()) & (df[second_event_name].isna())
        ].sort_values(by=first_event_name)

    sorted_Q_df = pd.concat([Q3_group, Q2_group, Q1_group]).reset_index(drop=True)

    for col in [first_event_name, second_event_name, third_event_name]:
        sorted_Q_df[col] = sorted_Q_df[col].apply(time_to_seconds)

    Q_result = sorted_Q_df[
        ["Driver", "Team", first_event_name, second_event_name, third_event_name]
    ]

    return Q_result


if __name__ == "__main__":
    year = 2026
    air_temp = 28.5
    pressure = 1013.6
    humidity = 60.0
    rainfall = 0
    circuit_id = "miami"

    drivers_data = {
        "Driver": [
            "PIA", "NOR", "VER", "RUS", "LEC", "ANT", "HAM", "ALB", "OCO", "STR",
            "HUL", "GAS", "BEA", "HAD", "SAI", "ALO", "LAW", "BOR", "LIN", "BOT",
            "PER", "COL",
        ]
    }

    teams_data = {
        "Team": [
            "McLaren", "McLaren", "Red Bull Racing", "Mercedes", "Ferrari", "Mercedes",
            "Ferrari", "Williams", "Haas F1 Team", "Aston Martin", "Audi", "Alpine",
            "Haas F1 Team", "Red Bull Racing", "Williams", "Aston Martin", "Racing Bulls",
            "Audi", "Racing Bulls", "Cadillac", "Cadillac", "Alpine",
        ]
    }

    df_driver = pd.DataFrame(drivers_data)
    df_teams = pd.DataFrame(teams_data)
    df = pd.concat([df_driver, df_teams], axis=1)

    print(df)
    print(
        get_prediction(
            year,
            air_temp,
            pressure,
            humidity,
            rainfall,
            "Q",
            4,
            df,
            19,
            5.412,
            True,
            circuit_id,
        )
    )
