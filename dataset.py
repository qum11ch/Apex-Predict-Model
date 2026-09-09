from typing import Optional

from main import load_single_file
import numpy as np
import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
ERGAST_DIR = ROOT / "ergast"


def add_quali_data():
    ergast = Ergast(result_type="pandas", auto_cast=True)

    gp_list = [
        "Monaco",
        "Barcelona",
        "Austria",
        "Britain",
        "Belgium",
        "Hungary",
        "Netherland",
        "Monza",
    ]

    current_year = 2026

    dataset_path = DATASET_DIR / f"QualiData_{current_year}_cleaned.csv"
    df = load_single_file(dataset_path)

    for gp in gp_list:
        last_year = 2027

        for year in range(2026, last_year):
            print(fastf1.get_event_schedule(2026))
            event1 = fastf1.get_session(year, gp, 1)

            if (event1.event["Location"] == gp) or (event1.event["Country"] == gp):
                gp_round = event1.event["RoundNumber"]
                race = ergast.get_race_schedule(season=year, round=gp_round)

                event1.load(laps=True, weather=True)

                current_season_quali = None
                if year > 2024:
                    current_season_quali = get_current_season_quali(
                        ergast, year, gp_round
                    )

                driver = get_driver_standings(ergast, year, gp_round)
                constructor = get_constructor_standings(ergast, year, gp_round)

                if (year == 2020) and (gp == "Imola"):
                    data = {"name": ["None"]}
                    event3 = pd.DataFrame(data)
                else:
                    event3 = fastf1.get_session(year, gp, 3)

                event4 = fastf1.get_session(year, gp, 4)
                event4.load(laps=True, weather=True)

                print(event4.load)

                if year == 2018:
                    event4_compounds = event4.laps.pick_compounds(
                        ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT"]
                    ).pick_quicklaps()
                else:
                    event4_compounds = event4.laps.pick_compounds(
                        ["SOFT", "MEDIUM"]
                    ).pick_quicklaps()

                event2_data = None
                event3_data = None
                event4_data = None

                if (year == 2021) or (year == 2022) or (year == 2023):
                    if event4.name != "Qualifying":
                        event2 = fastf1.get_session(year, gp, 2)
                        event2.load(laps=True, weather=True)

                        event2_compounds = event2.laps.pick_compounds(
                            ["SOFT", "MEDIUM"]
                        ).pick_quicklaps()

                        if event2_compounds.size != 0:
                            q1, q2, q3 = event2_compounds.split_qualifying_sessions()

                            if q1 is not None:
                                q1["Event"] = "Q1"
                            if q2 is not None:
                                q2["Event"] = "Q2"
                            if q3 is not None:
                                q3["Event"] = "Q3"

                            event2_total = pd.concat([q1, q2, q3], ignore_index=True)
                            event2_weather = event2.laps.get_weather_data().reset_index(
                                drop=True
                            )

                            event2_data = pd.concat(
                                [
                                    event2_total,
                                    event2_weather.loc[
                                        :, ~(event2_weather.columns == "Time")
                                    ],
                                ],
                                axis=1,
                            )
                            event2_data = event2_data.dropna(subset=["Driver"])
                            event2_data = add_season_stats(
                                event2_data,
                                driver,
                                constructor,
                                gp_round,
                                year,
                                current_season_quali,
                                event2,
                                ergast,
                            )

                    if year == 2023:
                        event3.load(laps=True, weather=True)
                        event3_compounds = event3.laps.pick_compounds(
                            ["SOFT", "MEDIUM"]
                        ).pick_quicklaps()

                        if event3_compounds.size != 0:
                            sq1, sq2, sq3 = event3_compounds.split_qualifying_sessions()

                            if sq1 is not None:
                                sq1["Event"] = "SQ1"
                            if sq2 is not None:
                                sq2["Event"] = "SQ2"
                            if sq3 is not None:
                                sq3["Event"] = "SQ3"

                            event3_total = pd.concat([sq1, sq2, sq3], ignore_index=True)
                            event3_weather = event3.laps.get_weather_data().reset_index(
                                drop=True
                            )

                            event3_data = pd.concat(
                                [
                                    event3_total,
                                    event3_weather.loc[
                                        :, ~(event3_weather.columns == "Time")
                                    ],
                                ],
                                axis=1,
                            )
                            event3_data = event3_data.dropna(subset=["Driver"])
                            event3_data = add_season_stats(
                                event3_data,
                                driver,
                                constructor,
                                gp_round,
                                year,
                                current_season_quali,
                                event3,
                                ergast,
                            )

                    if (event2_data is not None) or (event3_data is not None):
                        frames = [event2_data, event3_data]
                    elif event2_data is not None:
                        frames = [event2_data]
                    else:
                        frames = None
                else:
                    if event4_compounds.size != 0:
                        q1, q2, q3 = event4_compounds.split_qualifying_sessions()

                        if q1 is not None:
                            q1["Event"] = "Q1"
                        if q2 is not None:
                            q2["Event"] = "Q2"
                        if q3 is not None:
                            q3["Event"] = "Q3"

                        event4_total = pd.concat([q1, q2, q3], ignore_index=True)
                        event4_weather = event4.laps.get_weather_data().reset_index(
                            drop=True
                        )

                        event4_data = pd.concat(
                            [
                                event4_total,
                                event4_weather.loc[
                                    :, ~(event4_weather.columns == "Time")
                                ],
                            ],
                            axis=1,
                        )
                        event4_data = event4_data.dropna(subset=["Driver"])
                        event4_data = add_season_stats(
                            event4_data,
                            driver,
                            constructor,
                            gp_round,
                            year,
                            current_season_quali,
                            event4,
                            ergast,
                        )

                    if event4_data is not None:
                        frames = [event4_data]
                    else:
                        frames = None

                if (year == 2020) and (gp == "Imola"):
                    reg = "Practice 3"
                else:
                    reg = event3.name

                if reg != "Practice 3":
                    event2 = fastf1.get_session(year, gp, 2)
                    event2.load(laps=True, weather=True)

                    event2_compounds = event2.laps.pick_compounds(
                        ["SOFT", "MEDIUM"]
                    ).pick_quicklaps()

                    if event2_compounds.size != 0:
                        sq1, sq2, sq3 = event2_compounds.split_qualifying_sessions()

                        if sq1 is not None:
                            sq1["Event"] = "SQ1"
                        if sq2 is not None:
                            sq2["Event"] = "SQ2"
                        if sq3 is not None:
                            sq3["Event"] = "SQ3"

                        event2_total = pd.concat([sq1, sq2, sq3], ignore_index=True)
                        event2_weather = event2.laps.get_weather_data().reset_index(
                            drop=True
                        )

                        event2_data = pd.concat(
                            [
                                event2_total,
                                event2_weather.loc[
                                    :, ~(event2_weather.columns == "Time")
                                ],
                            ],
                            axis=1,
                        )
                        event2_data = event2_data.dropna(subset=["Driver"])
                        event2_data = add_season_stats(
                            event2_data,
                            driver,
                            constructor,
                            gp_round,
                            year,
                            current_season_quali,
                            event2,
                            ergast,
                        )

                    if (event2_data is not None) or (event4_data is not None):
                        frames = [event2_data, event4_data]
                    elif event4_data is not None:
                        frames = [event4_data]
                    else:
                        frames = None

                if frames is not None:
                    joined_data = pd.concat(frames).reset_index(drop=True)
                    joined_data = joined_data.query(
                        'Deleted == False & TrackStatus == "1"'
                    )
                    joined_data = joined_data[
                        [
                            "Driver",
                            "Team",
                            "LapTime",
                            "AirTemp",
                            "Humidity",
                            "Pressure",
                            "Rainfall",
                            "TrackTemp",
                            "DriverPoints",
                            "TeamPoints",
                            "DriverAvgQualiPos",
                            "TeamAvgQualiPos",
                            "Event",
                        ]
                    ]

                    circuits_len_df = get_circuits_len()
                    joined_data["LapTime"] = joined_data["LapTime"].dt.total_seconds()
                    joined_data["CircuitId"] = race.circuitId.values[0]
                    joined_data["Year"] = year
                    joined_data["IsStreetCircuit"] = is_street_circuit(
                        race.circuitId.values[0]
                    )
                    joined_data["F1Era"] = f1_era(year)
                    joined_data["CircuitCorners"] = (
                        event4.get_circuit_info().corners["Number"].values[-1]
                    )
                    joined_data["CircuitLength"] = circuits_len_df[
                        circuits_len_df["Circuit"] == race.circuitId.values[0]
                    ]["LapLength_km"].item()

                    joined_data["TrackAirDiff"] = None
                    joined_data["IsHighHumidity"] = None
                    joined_data["DriverQualiPace"] = None
                    joined_data["DriverSeasons"] = None
                    joined_data["TeamPointsContribution"] = None
                    joined_data["CarEngine"] = None

                    drivers_season_df = get_driver_seasons(season=year)
                    engine_season_df = get_engine_seasons(season=year)

                    for i, row in joined_data.iterrows():
                        air_temp = row["AirTemp"]
                        track_temp = row["TrackTemp"]
                        humidity = row["Humidity"]
                        driver_avg_quali = row["DriverAvgQualiPos"]
                        team_avg_quali = row["TeamAvgQualiPos"]
                        driver_code = row["Driver"]
                        driver_points = row["DriverPoints"]
                        team_points = row["TeamPoints"]
                        team_name = row["Team"]

                        track_air_diff = round((track_temp - air_temp), 2)
                        is_high_humidity = humidity > 70.0

                        if driver_avg_quali != 0.0:
                            driver_quali_pace = round(
                                (team_avg_quali - driver_avg_quali), 2
                            )
                        else:
                            driver_quali_pace = 0.0

                        if team_points != 0.0:
                            team_points_contribution = round(
                                (driver_points / team_points), 2
                            )
                        else:
                            team_points_contribution = 0.5

                        joined_data.at[i, "TeamPointsContribution"] = (
                            team_points_contribution
                        )
                        joined_data.at[i, "TrackAirDiff"] = track_air_diff
                        joined_data.at[i, "IsHighHumidity"] = is_high_humidity
                        joined_data.at[i, "DriverQualiPace"] = driver_quali_pace

                        print(driver_code)
                        print(drivers_season_df)
                        seasons_count = drivers_season_df[
                            drivers_season_df["driverCode"] == driver_code
                        ]["seasonsCount"].values

                        car_engine = engine_season_df[
                            engine_season_df["teamName"] == team_name
                        ]["engine"].values

                        if car_engine.size != 0:
                            joined_data.at[i, "CarEngine"] = car_engine.item()
                        else:
                            joined_data.at[i, "CarEngine"] = None

                        if seasons_count.size != 0:
                            joined_data.at[i, "DriverSeasons"] = seasons_count.item()
                        else:
                            k = year - 1
                            while k > 2017:
                                prev_drivers_season_df = get_driver_seasons(season=k)
                                new_seasons_count = prev_drivers_season_df[
                                    prev_drivers_season_df["driverCode"] == driver_code
                                ]["seasonsCount"].values
                                if new_seasons_count.size != 0:
                                    joined_data.at[i, "DriverSeasons"] = (
                                        new_seasons_count.item()
                                    )
                                    break
                                else:
                                    k -= 1
                                if k == 2017:
                                    joined_data.at[i, "DriverSeasons"] = 0.0

                    final_csv = pd.concat([df, joined_data])
                    df = final_csv

    new_dataset_path = DATASET_DIR / f"QualiData_{current_year}.csv"
    df.to_csv(new_dataset_path, index=False)


def get_current_season_quali(ergast: Ergast, year: int, race_round: int) -> pd.DataFrame:
    new_round = race_round - 1
    quali_df = pd.DataFrame(columns=["driverCode", "constructorId", "position"])

    while new_round > 0:
        quali_res = ergast.get_qualifying_results(
            season=year, round=new_round, result_type="pandas"
        ).content[0]

        results = pd.DataFrame(columns=["driverCode", "constructorId", "position"])
        results["driverCode"] = quali_res["driverCode"]
        results["constructorId"] = quali_res["constructorId"]
        results["position"] = quali_res["position"]

        quali_df = pd.concat([quali_df, results])
        new_round -= 1

    quali_df = quali_df.reset_index(drop=True)
    return quali_df


def add_season_stats(df: pd.DataFrame,
                     driver: pd.DataFrame,
                     constructor: pd.DataFrame,
                     race_round: int,
                     year: int,
                     current_season_qualis: Optional[pd.DataFrame],
                     event,
                     ergast: Ergast,
                     ) -> pd.DataFrame:

    drivers_df = load_single_file(ERGAST_DIR / "drivers.csv")
    qualifying_df = load_single_file(ERGAST_DIR / "qualifying.csv")
    races_df = load_single_file(ERGAST_DIR / "races.csv")
    constructors_df = load_single_file(ERGAST_DIR / "constructors.csv")

    df["DriverPoints"] = 0.0
    df["TeamPoints"] = 0.0
    df["DriverAvgQualiPos"] = 0.0
    df["TeamAvgQualiPos"] = 0.0

    for i, row in df.iterrows():
        driver_code = row["Driver"]
        team_name = row["Team"]

        driver_df = driver[driver["Driver"] == driver_code]
        driver_id = driver_df["DriverId"].astype(str).values
        driver_id_df = drivers_df[
            drivers_df["driverRef"].isin(driver_id)
        ]["driverId"].values

        if race_round == 1:
            current_season = ergast.get_driver_standings(
                season=year, round=race_round
            ).content[0]
            team_id = current_season[current_season["driverCode"] == driver_code][
                "constructorIds"
            ].values[0]
            team_id = np.stack(team_id, axis=0)
        else:
            team_id = driver[driver["Driver"] == driver_code]["TeamId"].values

        if team_id.size == 0:
            if (year == 2020) and (race_round == 4) and (driver_code == "HUL"):
                driver_team = "Racing Point"
            else:
                driver_team = df[df["Driver"] == driver_code]["Team"].values[0]

            event_laps = event.laps
            other_drivers = event_laps[event_laps["Team"] == driver_team]
            other_driver_code = other_drivers[
                other_drivers["Driver"] != driver_code
            ]["Driver"].values[0]

            team_id = driver[driver["Driver"] == other_driver_code]["TeamId"].values

            if team_id.size == 0 and team_name == "Red Bull Racing":
                team_id = np.array(["red_bull"])

        team_df = constructor[constructor["Constructor"].isin(team_id)]
        prev_constructor_standings = get_constructor_standings(ergast, year, 1)
        prev_team_df = prev_constructor_standings[
            prev_constructor_standings["Constructor"].isin(team_id)
        ]

        if prev_team_df.size == 0:
            prev_team_id_val = prev_team_id(year, team_id[0])
            if prev_team_id_val is not None:
                prev_team_df = prev_constructor_standings[
                    prev_constructor_standings["Constructor"] == prev_team_id_val
                ]

        if team_df.size != 0:
            df.at[i, "TeamPoints"] = float(team_df["Points"].values[0])
        else:
            prev_team_id_val = prev_team_id(year, team_id[0])
            if prev_team_id_val is not None:
                team_df = constructor[constructor["Constructor"] == prev_team_id_val]
                df.at[i, "TeamPoints"] = float(team_df["Points"].values[0])
            else:
                df.at[i, "TeamPoints"] = 0.0

        if driver_df.size != 0:
            df.at[i, "DriverPoints"] = float(driver_df["Points"].values[0])
        else:
            df.at[i, "DriverPoints"] = 0.0

        driver_quali_sum = 0.0
        team_quali_sum = 0.0
        prev_season_driver_quali_sum = 0.0
        prev_season_team_quali_sum = 0.0

        new_round = race_round - 1
        race_count_driver = race_round - 1
        race_count_team = race_round - 1

        prev_season_race_count_driver = races_df.query(
            f"year == {year - 1}"
        )["raceId"].values.size
        prev_season_race_count_team = races_df.query(
            f"year == {year - 1}"
        )["raceId"].values.size

        if race_round == 1:
            race_count_driver = races_df.query(f"year == {year - 1}")[
                "raceId"
            ].values.size
            race_count_team = races_df.query(f"year == {year - 1}")[
                "raceId"
            ].values.size

        if year < 2025:
            if race_round == 1:
                constructor_id_df = constructors_df[
                    constructors_df["constructorRef"].isin(team_df["Constructor"])
                ]["constructorId"].values

                race_id = races_df.query(f"year == {year - 1}")["raceId"].values

                for k in range(race_id.size):
                    driver_grid_pos = qualifying_df.query(
                        f"raceId == {race_id[k]} & driverId == {driver_id_df}"
                    )["position"].values

                    team_grid_pos = qualifying_df.query(
                        f"raceId == {race_id[k]} & constructorId == {constructor_id_df}"
                    )["position"].values

                    if driver_grid_pos.size == 0:
                        pos_to_sum_driver = 0.0
                        race_count_driver -= 1
                    else:
                        pos_to_sum_driver = driver_grid_pos.astype(float).sum()

                    if team_grid_pos.size == 0:
                        pos_to_sum_team = 0.0
                        race_count_team -= 1
                    else:
                        pos_to_sum_team = team_grid_pos.astype(float).sum()

                    driver_quali_sum += pos_to_sum_driver
                    team_quali_sum += pos_to_sum_team

            else:
                while new_round > 0:
                    constructor_id_df = constructors_df[
                        constructors_df["constructorRef"].isin(team_df["Constructor"])
                    ]["constructorId"].values

                    race_id = races_df.query(
                        f"year == {year} & round == {new_round}"
                    )["raceId"].values

                    driver_grid_pos = qualifying_df.query(
                        f"raceId == {race_id} & driverId == {driver_id_df}"
                    )["position"].values

                    team_grid_pos = qualifying_df.query(
                        f"raceId == {race_id} & constructorId == {constructor_id_df}"
                    )["position"].values

                    if driver_grid_pos.size == 0:
                        pos_to_sum_driver = 0.0
                        race_count_driver -= 1
                    else:
                        pos_to_sum_driver = driver_grid_pos.astype(float).item()

                    if team_grid_pos.size == 0:
                        pos_to_sum_team = 0.0
                        race_count_team -= 1
                    else:
                        pos_to_sum_team = team_grid_pos.astype(float).sum()

                    driver_quali_sum += pos_to_sum_driver
                    team_quali_sum += pos_to_sum_team
                    new_round -= 1

        else:
            if race_round != 1:
                driver_qualis = current_season_qualis.query(
                    f'driverCode == "{driver_code}"'
                )["position"].values

                driver_quali_count = driver_qualis.size

                team_qualis = current_season_qualis.loc[
                    current_season_qualis["constructorId"].isin(team_id)
                ]["position"].values
                team_quali_count = team_qualis.size

                if driver_quali_count != 0:
                    df.at[i, "DriverAvgQualiPos"] = round(driver_qualis.mean(), 2)
                else:
                    df.at[i, "DriverAvgQualiPos"] = 0.0

                if team_quali_count != 0:
                    df.at[i, "TeamAvgQualiPos"] = round(team_qualis.mean(), 2)
                else:
                    df.at[i, "TeamAvgQualiPos"] = 0.0

        if year < 2025:
            if race_count_driver != 0:
                avg_grid_driver = driver_quali_sum / race_count_driver
                df.at[i, "DriverAvgQualiPos"] = round(avg_grid_driver, 2)
            else:
                df.at[i, "DriverAvgQualiPos"] = 0.0

            if race_count_team != 0:
                avg_grid_team = team_quali_sum / (race_count_team * 2)
                df.at[i, "TeamAvgQualiPos"] = round(avg_grid_team, 2)
            else:
                df.at[i, "TeamAvgQualiPos"] = 0.0

        constructor_id_df = constructors_df[
            constructors_df["constructorRef"].isin(prev_team_df["Constructor"])
        ]["constructorId"].values
        prev_season_races_id = races_df.query(f"year == {year - 1}")["raceId"].values

        for k in range(prev_season_races_id.size):
            prev_season_driver_grid_pos = qualifying_df.query(
                f"raceId == {prev_season_races_id[k]} & driverId == {driver_id_df}"
            )["position"].values

            prev_season_team_grid_pos = qualifying_df.query(
                f"raceId == {prev_season_races_id[k]} & constructorId == {constructor_id_df}"
            )["position"].values

            if prev_season_driver_grid_pos.size == 0:
                prev_season_grid = 0.0
                prev_season_race_count_driver -= 1
            else:
                prev_season_grid = prev_season_driver_grid_pos.astype(float).sum()

            if prev_season_team_grid_pos.size == 0:
                prev_season_team_grid = 0.0
                prev_season_race_count_team -= 1
            else:
                prev_season_team_grid = prev_season_team_grid_pos.astype(float).sum()

            prev_season_driver_quali_sum += prev_season_grid
            prev_season_team_quali_sum += prev_season_team_grid

        if prev_season_race_count_driver != 0:
            prev_season_avg_grid_driver = (
                prev_season_driver_quali_sum / prev_season_race_count_driver
            )
            if (race_round == 1) and (year > 2024):
                df.at[i, "DriverAvgQualiPos"] = round(prev_season_avg_grid_driver, 2)
            elif (race_round == 1) and (year > 2024):
                df.at[i, "DriverAvgQualiPos"] = 0.0

        if prev_season_race_count_team != 0:
            prev_season_avg_grid_team = (
                prev_season_team_quali_sum / (prev_season_race_count_team * 2)
            )
            if (race_round == 1) and (year > 2024):
                df.at[i, "TeamAvgQualiPos"] = round(prev_season_avg_grid_team, 2)
            elif (race_round == 1) and (year > 2024):
                df.at[i, "TeamAvgQualiPos"] = 0.0

    return df


def get_driver_standings(ergast: Ergast, season: int, race_round: int) -> pd.DataFrame:
    if race_round > 1:
        standings = ergast.get_driver_standings(season=season, round=race_round - 1)
    else:
        standings = ergast.get_driver_standings(season=season - 1)

    driver = standings.content[0]
    driver = driver[["driverCode", "points", "wins", "constructorIds", "driverId"]]
    driver.columns = ["Driver", "Points", "Wins", "TeamId", "DriverId"]

    for i, row in driver.iterrows():
        driver.at[i, "TeamId"] = row["TeamId"][-1]

    return driver


def get_constructor_standings(ergast: Ergast, season: int, race_round: int) -> pd.DataFrame:
    if race_round > 1:
        standings = ergast.get_constructor_standings(
            season=season, round=race_round - 1
        )
    else:
        standings = ergast.get_constructor_standings(season=season - 1)

    constructor = standings.content[0]
    constructor = constructor[["constructorId", "points", "wins"]]
    constructor.columns = ["Constructor", "Points", "Wins"]

    return constructor


def prev_team_id(season: int, team_id: str) -> Optional[str]:
    mapping = {
        (2019, "racing_point"): "force_india",
        (2020, "alphatauri"): "toro_rosso",
        (2021, "alpine"): "renault",
        (2021, "aston_martin"): "racing_point",
        (2024, "rb"): "alphatauri",
        (2024, "sauber"): "alfa",
        (2026, "audi"): "sauber",
    }
    return mapping.get((season, team_id))


def f1_era(season: int) -> str:
    if 2017 <= season <= 2021:
        return 'WideAero'
    elif 2022 <= season <= 2025:
        return 'GroundEffect'
    elif 2026 <= season:
        return 'ActiveAero'
    else:
        return 'GroundEffect'


def is_street_circuit(circuitId: str) -> bool:
    street_circuits = {
        "albert_park",
        "baku",
        "monaco",
        "villeneuve",
        "jeddah",
        "vegas",
        "miami",
        "marina_bay",
        "madring",
    }
    return circuitId in street_circuits


def get_circuits_len() -> pd.DataFrame:
    data = {
        'Circuit': [
            'albert_park', 'shanghai', 'suzuka', 'bahrain', 'jeddah', 'miami', 'imola', 'monaco', 'catalunya',
            'villeneuve', 'red_bull_ring', 'silverstone', 'spa', 'hungaroring', 'zandvoort', 'monza', 'baku',
            'marina_bay', 'americas', 'rodriguez', 'interlagos', 'vegas', 'losail', 'yas_marina', 'madring', 'sepang'],
        'LapLength_km': [5.278, 5.451, 5.807, 5.412, 6.174, 5.412, 4.909, 3.337, 4.657, 4.361, 4.318, 5.891, 7.004,
                         4.381, 4.259, 5.793, 6.003, 4.94, 5.513, 4.304, 4.309, 6.201, 5.419, 5.281, 5.474, 5.543]
    }
    df = pd.DataFrame(data)
    return df


def get_driver_seasons(season: int) -> pd.DataFrame:
    data_by_year = {
        2018: {
            "driverCode": [
                "ALO",
                "BOT",
                "ERI",
                "GAS",
                "GRO",
                "HAM",
                "HAR",
                "HUL",
                "LEC",
                "MAG",
                "OCO",
                "PER",
                "RAI",
                "RIC",
                "SAI",
                "SIR",
                "STR",
                "VAN",
                "VER",
                "VET",
            ],
            "seasonsCount": [
                16,
                5,
                4,
                1,
                7,
                11,
                1,
                7,
                0,
                3,
                2,
                7,
                15,
                7,
                3,
                0,
                1,
                1,
                3,
                11,
            ],
        },
        2019: {
            "driverCode": [
                "HAM",
                "BOT",
                "VER",
                "LEC",
                "VET",
                "SAI",
                "GAS",
                "ALB",
                "RIC",
                "PER",
                "NOR",
                "RAI",
                "KVY",
                "HUL",
                "STR",
                "MAG",
                "GIO",
                "GRO",
                "KUB",
                "RUS",
            ],
            "seasonsCount": [
                12,
                6,
                4,
                1,
                12,
                4,
                2,
                0,
                8,
                8,
                0,
                16,
                4,
                8,
                2,
                4,
                1,
                8,
                5,
                0,
            ],
        },
        2020: {
            "driverCode": [
                "HAM",
                "BOT",
                "VER",
                "PER",
                "RIC",
                "SAI",
                "ALB",
                "LEC",
                "NOR",
                "GAS",
                "STR",
                "OCO",
                "VET",
                "KVY",
                "HUL",
                "RAI",
                "GIO",
                "RUS",
                "GRO",
                "MAG",
                "LAT",
                "AIT",
                "FIT",
            ],
            "seasonsCount": [
                13,
                7,
                5,
                9,
                9,
                5,
                1,
                2,
                1,
                3,
                3,
                4,
                13,
                5,
                9,
                17,
                2,
                1,
                9,
                5,
                0,
                0,
                0,
            ],
        },
        2021: {
            "driverCode": [
                "VER",
                "HAM",
                "BOT",
                "PER",
                "SAI",
                "NOR",
                "LEC",
                "RIC",
                "GAS",
                "ALO",
                "OCO",
                "VET",
                "STR",
                "TSU",
                "RUS",
                "RAI",
                "LAT",
                "GIO",
                "MSC",
                "KUB",
                "MAZ",
            ],
            "seasonsCount": [
                6,
                14,
                8,
                10,
                6,
                2,
                3,
                10,
                4,
                17,
                5,
                14,
                4,
                0,
                2,
                18,
                1,
                3,
                0,
                7,
                0,
            ],
        },
        2022: {
            "driverCode": [
                "VER",
                "LEC",
                "PER",
                "RUS",
                "SAI",
                "HAM",
                "NOR",
                "OCO",
                "ALO",
                "BOT",
                "RIC",
                "VET",
                "MAG",
                "GAS",
                "STR",
                "MSC",
                "TSU",
                "ZHO",
                "ALB",
                "LAT",
                "DEV",
                "HUL",
            ],
            "seasonsCount": [
                7,
                4,
                11,
                3,
                7,
                15,
                3,
                6,
                18,
                9,
                11,
                15,
                7,
                5,
                5,
                1,
                1,
                0,
                2,
                2,
                0,
                10,
            ],
        },
        2023: {
            "driverCode": [
                "VER",
                "PER",
                "HAM",
                "ALO",
                "LEC",
                "NOR",
                "SAI",
                "RUS",
                "PIA",
                "STR",
                "GAS",
                "OCO",
                "ALB",
                "TSU",
                "BOT",
                "HUL",
                "RIC",
                "ZHO",
                "MAG",
                "LAW",
                "SAR",
                "DEV",
            ],
            "seasonsCount": [
                8,
                12,
                16,
                19,
                5,
                4,
                8,
                4,
                0,
                6,
                6,
                7,
                4,
                2,
                10,
                10,
                12,
                1,
                8,
                0,
                0,
                0,
            ],
        },
        2024: {
            "driverCode": [
                "VER",
                "NOR",
                "LEC",
                "PIA",
                "SAI",
                "RUS",
                "HAM",
                "PER",
                "ALO",
                "GAS",
                "HUL",
                "TSU",
                "STR",
                "OCO",
                "MAG",
                "ALB",
                "RIC",
                "BEA",
                "COL",
                "ZHO",
                "LAW",
                "BOT",
                "SAR",
                "DOO",
            ],
            "seasonsCount": [
                9,
                5,
                6,
                1,
                9,
                5,
                17,
                13,
                20,
                7,
                13,
                3,
                7,
                8,
                9,
                4,
                13,
                0,
                0,
                2,
                1,
                11,
                1,
                0,
            ],
        },
        2025: {
            "driverCode": [
                "PIA",
                "NOR",
                "VER",
                "RUS",
                "LEC",
                "ANT",
                "HAM",
                "ALB",
                "OCO",
                "STR",
                "HUL",
                "GAS",
                "BEA",
                "HAD",
                "SAI",
                "TSU",
                "ALO",
                "LAW",
                "DOO",
                "BOR",
            ],
            "seasonsCount": [
                2,
                6,
                10,
                6,
                7,
                5,
                18,
                6,
                9,
                8,
                12,
                8,
                1,
                1,
                10,
                4,
                21,
                1,
                0,
                0,
            ],
        },
        2026: {
            "driverCode": [
                "PIA",
                "NOR",
                "VER",
                "RUS",
                "LEC",
                "ANT",
                "HAM",
                "ALB",
                "OCO",
                "STR",
                "HUL",
                "GAS",
                "BEA",
                "HAD",
                "SAI",
                "ALO",
                "LAW",
                "BOR",
                "LIN",
                "BOT",
                "PER",
                "COL",
            ],
            "seasonsCount": [
                3,
                8,
                11,
                7,
                8,
                1,
                19,
                6,
                10,
                9,
                15,
                9,
                1,
                1,
                11,
                22,
                2,
                1,
                0,
                12,
                13,
                2,
            ],
        },
    }

    data = data_by_year.get(season, {"driverCode": [], "seasonsCount": []})
    return pd.DataFrame(data)


def get_engine_seasons(season: int) -> pd.DataFrame:
    """
    Возвращает DataFrame с двигателями команд по сезону.
    """
    data_by_year = {
        2018: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Renault",
                "Haas F1 Team",
                "McLaren",
                "Force India",
                "Racing Point",
                "Toro Rosso",
                "Sauber",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Renault",
                "Renault",
                "Ferrari",
                "Renault",
                "Mercedes",
                "Mercedes",
                "Honda",
                "Ferrari",
                "Mercedes",
            ],
        },
        2019: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Renault",
                "Haas F1 Team",
                "McLaren",
                "Racing Point",
                "Toro Rosso",
                "Alfa Romeo Racing",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda",
                "Renault",
                "Ferrari",
                "Renault",
                "Mercedes",
                "Honda",
                "Ferrari",
                "Mercedes",
            ],
        },
        2020: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Renault",
                "Haas F1 Team",
                "McLaren",
                "Racing Point",
                "AlphaTauri",
                "Alfa Romeo Racing",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda",
                "Renault",
                "Ferrari",
                "Renault",
                "Mercedes",
                "Honda",
                "Ferrari",
                "Mercedes",
            ],
        },
        2021: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "AlphaTauri",
                "Alfa Romeo Racing",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda",
                "Renault",
                "Ferrari",
                "Mercedes",
                "Mercedes",
                "Honda",
                "Ferrari",
                "Mercedes",
            ],
        },
        2022: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "AlphaTauri",
                "Alfa Romeo",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda RBPT",
                "Renault",
                "Ferrari",
                "Mercedes",
                "Mercedes",
                "Honda RBPT",
                "Ferrari",
                "Mercedes",
            ],
        },
        2023: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "AlphaTauri",
                "Alfa Romeo",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda RBPT",
                "Renault",
                "Ferrari",
                "Mercedes",
                "Mercedes",
                "Honda RBPT",
                "Ferrari",
                "Mercedes",
            ],
        },
        2024: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "RB",
                "Kick Sauber",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda RBPT",
                "Renault",
                "Ferrari",
                "Mercedes",
                "Mercedes",
                "Honda RBPT",
                "Ferrari",
                "Mercedes",
            ],
        },
        2025: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "Racing Bulls",
                "Kick Sauber",
                "Williams",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Honda RBPT",
                "Renault",
                "Ferrari",
                "Mercedes",
                "Mercedes",
                "Honda RBPT",
                "Ferrari",
                "Mercedes",
            ],
        },
        2026: {
            "teamName": [
                "Mercedes",
                "Ferrari",
                "Red Bull Racing",
                "Alpine",
                "Haas F1 Team",
                "McLaren",
                "Aston Martin",
                "Racing Bulls",
                "Audi",
                "Williams",
                "Cadillac",
            ],
            "engine": [
                "Mercedes",
                "Ferrari",
                "Ford RBPT",
                "Mercedes",
                "Ferrari",
                "Mercedes",
                "Honda",
                "Ford RBPT",
                "Audi",
                "Mercedes",
                "Ferrari",
            ],
        },
    }

    data = data_by_year.get(season, {"teamName": [], "engine": []})
    return pd.DataFrame(data)


def dataset_main():
    add_quali_data()


if __name__ == '__main__':
    dataset_main()
