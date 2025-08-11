import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from fastf1.ergast import Ergast
from catboost import CatBoostRegressor

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ergast = Ergast(result_type='pandas', auto_cast=True)

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


def time_to_seconds(seconds):
    if pd.isna(seconds):
        return np.nan
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes}:{remaining_seconds:06.3f}"


def test_data(year, airTemp, pressure, humidity, rainfall, event, gpRound, df, circuitCorners,
              circuitLength, isStreetCircuit):

    currentSeasonQuali = None
    if year > 2024:
        currentSeasonQuali = get_current_season_quali(year, gpRound)

    driver = get_driver_standings(year, gpRound)
    constructor = get_constructor_standings(year, gpRound)

    df = add_season_stats(df, driver, constructor, gpRound, year, currentSeasonQuali)

    first_event_name = event + "1"

    df['Year'] = year
    df['IsStreetCircuit'] = isStreetCircuit
    df['F1Era'] = f1_era(year)
    df['CircuitCorners'] = circuitCorners
    df['CircuitLength'] = circuitLength
    df['Event'] = first_event_name

    # TrackTemp predict
    tt_cb_model = CatBoostRegressor().load_model("models/tt_cat_boost_model.cbm")
    tt_preprocessor = load("preprocessors/tt_data_preprocessor.joblib")
    tt_y_scaler = load("preprocessors/tt_y_scaler.joblib")

    IsStreetCircuit = df['IsStreetCircuit'].values[0]
    circuitsCorners = df['CircuitCorners'].values[0]
    circuitLength = df['CircuitLength'].values[0]
    isHighHumidity = 0
    if humidity > 70.0:
        isHighHumidity = 1

    tt_data = {
        'Year': [year],
        'AirTemp': [airTemp],
        'Humidity': [humidity],
        'Pressure': [pressure],
        'Rainfall': [rainfall],
        'IsStreetCircuit': [IsStreetCircuit],
        'CircuitCorners': [circuitsCorners],
        'IsHighHumidity': [isHighHumidity],
        'CircuitLength': [circuitLength]
    }

    tt_df = pd.DataFrame(tt_data)
    tt_data = tt_preprocessor.transform(tt_df)

    trackTemp_normal = tt_cb_model.predict(tt_data)
    trackTemp = tt_y_scaler.inverse_transform(trackTemp_normal.reshape(-1, 1)).item()
    trackTemp = round(trackTemp, 1)
    trackAirDiff = trackTemp - airTemp
    trackAirDiff = round(trackAirDiff, 1)

    df['AirTemp'] = airTemp
    df['Humidity'] = humidity
    df['Pressure'] = pressure
    df['Rainfall'] = rainfall
    df['TrackTemp'] = trackTemp
    df['TrackAirDiff'] = trackAirDiff
    df['IsHighHumidity'] = isHighHumidity

    df['DriverQualiPace'] = None
    df['DriverSeasons'] = None
    df['TeamPointsContribution'] = None
    df['CarEngine'] = None

    driversSeasonDf = get_driver_seasons(season=year)
    engineSeasonDf = get_engine_seasons(season=year)

    for i, j in df.iterrows():
        driverAvgQuali = j['DriverAvgQualiPos']
        teamAvgQuali = j['TeamAvgQualiPos']
        driverCode = j['Driver']
        driverPoints = j['DriverPoints']
        teamPoints = j['TeamPoints']
        teamName = j['Team']

        if driverAvgQuali != 0.0:
            driverQualiPace = round((teamAvgQuali - driverAvgQuali), 2)
        else:
            driverQualiPace = 0.0

        if teamPoints != 0.0:
            teamPointsContribution = round((driverPoints / teamPoints), 2)
        else:
            teamPointsContribution = 0.5

        df.at[i, 'TeamPointsContribution'] = teamPointsContribution
        df.at[i, 'DriverQualiPace'] = driverQualiPace

        seasonsCount = driversSeasonDf[driversSeasonDf['driverCode'] == driverCode][
            'seasonsCount'].values
        carEngine = engineSeasonDf[engineSeasonDf['teamName'] == teamName]['engine'].values

        if carEngine.size != 0:
            df.at[i, 'CarEngine'] = carEngine.item()
        else:
            df.at[i, 'CarEngine'] = None

        if seasonsCount.size != 0:
            df.at[i, 'DriverSeasons'] = seasonsCount.item()
        else:
            k = year - 1
            while k > 2017:
                prevDriversSeasonDf = get_driver_seasons(season=k)
                newSeasonsCount = prevDriversSeasonDf[prevDriversSeasonDf['driverCode'] == driverCode][
                    'seasonsCount'].values
                if newSeasonsCount.size != 0:
                    df.at[i, 'DriverSeasons'] = newSeasonsCount.item()
                    break
                else:
                    k -= 1
            if k == 2017:
                df.at[i, 'DriverSeasons'] = 0.0

    #print(df)

    features = ['Driver', 'Team', 'Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
                'DriverPoints', 'TeamPoints', 'DriverAvgQualiPos', 'TeamAvgQualiPos', 'IsStreetCircuit', 'F1Era',
                'CircuitCorners', 'TrackAirDiff', 'IsHighHumidity', 'DriverQualiPace', 'DriverSeasons',
                'CircuitLength', 'TeamPointsContribution', 'CarEngine', 'Event']

    loaded_model = tf.keras.models.load_model("models/best_laptime_model.keras",
                                              custom_objects={'metrics>huber_loss': huber_loss})

    cb_model = CatBoostRegressor().load_model('models/cat_boost_model.cbm')

    Q1_X = df[features]

    preprocessor = load("preprocessors/data_preprocessor.joblib")
    y_scaler = load("preprocessors/y_scaler.joblib")

    cb_feature_names = preprocessor.get_feature_names_out().tolist()

    # Q1 Time
    Q1_processed = preprocessor.transform(Q1_X)
    cb_Q1_pred = cb_model.predict(Q1_processed).reshape(-1, 1)
    X_Q1_catnn = np.hstack([Q1_processed, cb_Q1_pred])

    Q1_pred_neuro = loaded_model.predict(X_Q1_catnn)

    Q1_pred_neuro_inversed = y_scaler.inverse_transform(Q1_pred_neuro)

    df[first_event_name] = Q1_pred_neuro_inversed

    # Q2 Time
    Q2_df = df.sort_values(by=[first_event_name]).head(15)

    second_event_name = event + "2"
    Q2_df['Event'] = second_event_name

    Q2_X = Q2_df[features]
    Q2_processed = preprocessor.transform(Q2_X)
    cb_Q2_preds = cb_model.predict(Q2_processed).reshape(-1, 1)
    X_Q2_catnn = np.hstack([Q2_processed, cb_Q2_preds])

    Q2_pred_neuro = loaded_model.predict(X_Q2_catnn)

    Q2_pred_neuro_inversed = y_scaler.inverse_transform(Q2_pred_neuro)

    Q2_df[second_event_name] = Q2_pred_neuro_inversed

    Q2_results = Q2_df.sort_values(by=[second_event_name])[['Driver', second_event_name]]
    Q2_results = Q2_results.reset_index(drop=True)

    # Q3 TIME
    Q3_df = Q2_df.sort_values(by=[second_event_name]).head(10)

    third_event_name = event + "3"
    Q3_df['Event'] = third_event_name

    Q3_x = Q3_df[features]
    Q3_processed = preprocessor.transform(Q3_x)
    cb_Q3_preds = cb_model.predict(Q3_processed).reshape(-1, 1)
    X_Q3_catnn = np.hstack([Q3_processed, cb_Q3_preds])

    Q3_pred_neuro = loaded_model.predict(X_Q3_catnn)

    Q3_pred_neuro_inversed = y_scaler.inverse_transform(Q3_pred_neuro)

    Q3_df[third_event_name] = Q3_pred_neuro_inversed

    Q3_result = Q3_df.sort_values(by=[third_event_name])[['Driver', third_event_name]]
    Q3_result = Q3_result.reset_index(drop=True)

    # Summary
    df[second_event_name] = None
    df[third_event_name] = None
    for i, j in df.iterrows():
        driverCode = j['Driver']
        Q2_time = Q2_results[Q2_results['Driver'] == driverCode][second_event_name].values
        Q3_time = Q3_result[Q3_result['Driver'] == driverCode][third_event_name].values

        if len(Q2_time) == 0:
            Q2_time = np.nan
        if len(Q3_time) == 0:
            Q3_time = np.nan

        df.at[i, second_event_name] = Q2_time
        df.at[i, third_event_name] = Q3_time

    Q3_group = df[df[third_event_name].notna()].sort_values(by=second_event_name)
    Q2_group = df[df[third_event_name].isna() & df[second_event_name].notna()].sort_values(by=second_event_name)
    Q1_group = df[df[third_event_name].isna() & df[second_event_name].isna()].sort_values(by=first_event_name)

    sorted_Q_df = pd.concat([Q3_group, Q2_group, Q1_group]).reset_index(drop=True)

    for col in [first_event_name, second_event_name, third_event_name]:
        sorted_Q_df[col] = sorted_Q_df[col].apply(time_to_seconds)

    Q_result = sorted_Q_df[['Driver', 'Team', first_event_name, second_event_name, third_event_name]]
    print('Quali results')
    print(Q_result)
    Q_res_dict = Q_result.to_dict()


def add_season_stats(df, driver, constructor, raceRound, year, currentSeasonQualis):
    drivers_df = pd.read_csv("ergast/drivers.csv")
    qualifying_df = pd.read_csv("ergast/qualifying.csv")
    races_df = pd.read_csv("ergast/races.csv")
    constructors_df = pd.read_csv("ergast/constructors.csv")

    df[['DriverPoints', 'TeamPoints', 'DriverAvgQualiPos', 'TeamAvgQualiPos']] = None

    for i, j in df.iterrows():
        driverCode = j['Driver']
        driverDf = driver[driver['Driver'] == driverCode]

        teamName = j['Team']

        # Для CVS-шника
        driverId = driverDf['DriverId'].astype('string').values
        driverId_df = drivers_df[drivers_df['driverRef'].isin(driverId)]['driverId'].values

        if raceRound == 1:
            currentSeason = ergast.get_driver_standings(season=year, round=raceRound).content[0]
            teamId = currentSeason[currentSeason['driverCode'] == driverCode]['constructorIds'].values[0]
            teamId = np.stack(teamId, axis=0)
        else:
            teamId = driver[driver['Driver'] == driverCode]['TeamId'].values

        if driverDf.size == 0:
            teamDriversDf = df[df['Team'] == teamName]
            otherDriverDf = teamDriversDf[teamDriversDf['Driver'] != driverCode]
            otherDriverCode = otherDriverDf['Driver'].item()

            if teamId.size == 0:
                teamId = driver[driver['Driver'] == otherDriverCode]['TeamId'].values
        # print(driverDf)

        teamDf = constructor[constructor['Constructor'].isin(teamId)]

        prevDriverStandings = get_driver_standings(year, 1)
        prevConstructorStandings = get_constructor_standings(year, 1)

        prevDriverDf = prevDriverStandings[prevDriverStandings['Driver'] == driverCode]
        prevTeamDf = prevConstructorStandings[prevConstructorStandings['Constructor'].isin(teamId)]

        if prevTeamDf.size == 0:
            prev_teamId = prev_team_id(year, teamId[0])
            if prev_teamId is not None:
                prevTeamDf = prevConstructorStandings[prevConstructorStandings['Constructor'] == prev_teamId]

        # Обычные данные по сезону
        if teamDf.size != 0:
            df.at[i, 'TeamPoints'] = teamDf['Points'].values.astype(float)

        else:
            prev_teamId = prev_team_id(year, teamId[0])
            if prev_teamId is not None:
                teamDf = constructor[constructor['Constructor'] == prev_teamId]
                df.at[i, 'TeamPoints'] = teamDf['Points'].values.astype(float)

            else:
                df.at[i, 'TeamPoints'] = 0.0

        if driverDf.size != 0:
            # Это уже у нас сохранение данных
            df.at[i, 'DriverPoints'] = driverDf['Points'].values.astype(float)

        else:
            # Это уже у нас сохранение данных
            df.at[i, 'DriverPoints'] = 0.0

        driver_quali_sum = 0.0
        team_quali_sum = 0.0
        prevSeasonDriver_quali_sum = 0.0
        prevSeasonTeam_quali_sum = 0.0

        new_round = raceRound - 1
        raceCountDriver = raceRound - 1
        raceCountTeam = raceRound - 1

        prevSeasonRaceCountDriver = races_df.query(f'year == {year - 1}')['raceId'].values.size
        prevSeasonRaceCountTeam = races_df.query(f'year == {year - 1}')['raceId'].values.size
        if raceRound == 1:
            raceCountDriver = races_df.query(f'year == {year - 1}')['raceId'].values.size
            raceCountTeam = races_df.query(f'year == {year - 1}')['raceId'].values.size

        if year < 2025:
            if raceRound == 1:
                constructorId_df = constructors_df[constructors_df['constructorRef'].isin(teamDf['Constructor'])][
                    'constructorId'].values

                raceId = races_df.query(f'year == {year - 1}')['raceId'].values

                for k in range(0, raceId.size):
                    driver_gridPos = qualifying_df.query(f'raceId == {raceId[k]} & driverId == {driverId_df}')[
                        'position'].values
                    team_gridPos = \
                        qualifying_df.query(f'raceId == {raceId[k]} & constructorId == {constructorId_df}')[
                            'position'].values

                    if driver_gridPos.size == 0:
                        pos_to_sum_driver = 0.0
                        raceCountDriver -= 1
                    else:
                        pos_to_sum_driver = driver_gridPos.astype(float).sum()

                    if team_gridPos.size == 0:
                        pos_to_sum_team = 0.0
                        raceCountTeam -= 1
                    else:
                        pos_to_sum_team = team_gridPos.astype(float).sum()

                    driver_quali_sum += pos_to_sum_driver
                    team_quali_sum += pos_to_sum_team

            else:
                while new_round > 0:
                    # Для CVS-шника
                    constructorId_df = constructors_df[constructors_df['constructorRef'].isin(teamDf['Constructor'])][
                        'constructorId'].values

                    raceId = races_df.query(f'year == {year} & round == {new_round}')['raceId'].values

                    driver_gridPos = qualifying_df.query(f'raceId == {raceId} & driverId == {driverId_df}')[
                        'position'].values

                    team_gridPos = qualifying_df.query(f'raceId == {raceId} & constructorId == {constructorId_df}')[
                        'position'].values

                    if driver_gridPos.size == 0:
                        pos_to_sum_driver = 0.0
                        raceCountDriver -= 1
                    else:
                        pos_to_sum_driver = driver_gridPos.astype(float).item()

                    if team_gridPos.size == 0:
                        pos_to_sum_team = 0.0
                        raceCountTeam -= 1
                    else:
                        pos_to_sum_team = team_gridPos.astype(float).sum()

                    driver_quali_sum += pos_to_sum_driver
                    team_quali_sum += pos_to_sum_team
                    new_round -= 1
        else:
            if raceRound != 1:
                driverQualis = currentSeasonQualis.query(f'driverCode == "{driverCode}"')['position'].values
                driver_quali_count = driverQualis.size
                driver_quali_sum = driverQualis.astype(float).sum()

                teamQualis = currentSeasonQualis.loc[currentSeasonQualis['constructorId'].isin(teamId)][
                    'position'].values
                team_quali_count = teamQualis.size
                team_quali_sum = teamQualis.astype(float).sum()

                if driver_quali_count != 0:
                    avg_grid_driver = driver_quali_sum / driver_quali_count
                    df.at[i, 'DriverAvgQualiPos'] = round(avg_grid_driver, 2)
                else:
                    df.at[i, 'DriverAvgQualiPos'] = 0.0

                if team_quali_count != 0:
                    avg_grid_team = team_quali_sum / team_quali_count
                    df.at[i, 'TeamAvgQualiPos'] = round(avg_grid_team, 2)
                else:
                    df.at[i, 'TeamAvgQualiPos'] = 0.0

        if year < 2025:
            if raceCountDriver != 0:
                avg_grid_driver = driver_quali_sum / raceCountDriver
                df.at[i, 'DriverAvgQualiPos'] = round(avg_grid_driver, 2)
            else:
                df.at[i, 'DriverAvgQualiPos'] = 0.0

            if raceCountTeam != 0:
                avg_grid_team = team_quali_sum / (raceCountTeam * 2)
                df.at[i, 'TeamAvgQualiPos'] = round(avg_grid_team, 2)
            else:
                df.at[i, 'TeamAvgQualiPos'] = 0.0

        # if raceRound != 1:
        # Прошлый сезон
        constructorId_df = constructors_df[constructors_df['constructorRef'].isin(prevTeamDf['Constructor'])][
            'constructorId'].values
        prevSeasonRacesId = races_df.query(f'year == {year - 1}')['raceId'].values

        for k in range(0, prevSeasonRacesId.size):
            prevSeasonDriver_gridPos = \
                qualifying_df.query(f'raceId == {prevSeasonRacesId[k]} & driverId == {driverId_df}')[
                    'position'].values
            prevSeasonTeam_gridPos = \
                qualifying_df.query(
                    f'raceId == {prevSeasonRacesId[k]} & constructorId == {constructorId_df}')[
                    'position'].values

            if prevSeasonDriver_gridPos.size == 0:
                prevSeasonGrid = 0.0
                prevSeasonRaceCountDriver -= 1
            else:
                prevSeasonGrid = prevSeasonDriver_gridPos.astype(float).sum()

            if prevSeasonTeam_gridPos.size == 0:
                prevSeasonTeamGrid = 0.0
                prevSeasonRaceCountTeam -= 1
            else:
                prevSeasonTeamGrid = prevSeasonTeam_gridPos.astype(float).sum()

            prevSeasonDriver_quali_sum += prevSeasonGrid
            prevSeasonTeam_quali_sum += prevSeasonTeamGrid

        if prevSeasonRaceCountDriver != 0:
            prevSeasonAvg_grid_driver = prevSeasonDriver_quali_sum / prevSeasonRaceCountDriver
            if (raceRound == 1) & (year > 2024):
                df.at[i, 'DriverAvgQualiPos'] = round(prevSeasonAvg_grid_driver, 2)
        else:
            if (raceRound == 1) & (year > 2024):
                df.at[i, 'DriverAvgQualiPos'] = 0.0

        if prevSeasonRaceCountTeam != 0:
            prevSeasonAvg_grid_team = prevSeasonTeam_quali_sum / (prevSeasonRaceCountTeam * 2)
            if (raceRound == 1) & (year > 2024):
                df.at[i, 'TeamAvgQualiPos'] = round(prevSeasonAvg_grid_team, 2)
        else:
            if (raceRound == 1) & (year > 2024):
                df.at[i, 'TeamAvgQualiPos'] = 0.0

    return df


def get_driver_standings(season, raceRound):
    if raceRound > 1:
        driverStandings = ergast.get_driver_standings(season=season, round=raceRound - 1)
    else:
        driverStandings = ergast.get_driver_standings(season=season - 1)

    driver = driverStandings.content[0]
    driver = driver[['driverCode', 'points', 'wins', 'constructorIds', 'driverId']]
    driver.columns = ['Driver', 'Points', 'Wins', 'TeamId', 'DriverId']

    for i, j in driver.iterrows():
        driverTeamId = j['TeamId'][-1]
        driver.at[i, 'TeamId'] = driverTeamId

    return driver


def get_constructor_standings(season, raceRound):
    if raceRound > 1:
        constructorStandings = ergast.get_constructor_standings(season=season, round=raceRound - 1)
    else:
        constructorStandings = ergast.get_constructor_standings(season=season - 1)

    constructor = constructorStandings.content[0]
    constructor = constructor[['constructorId', 'points', 'wins']]
    constructor.columns = ['Constructor', 'Points', 'Wins']

    return constructor


def get_current_season_quali(year, raceRound):
    new_round = raceRound - 1
    quali_df = pd.DataFrame(columns=['driverCode', 'constructorId', 'position'])

    while new_round > 0:
        qualiRes = ergast.get_qualifying_results(season=year, round=new_round, result_type='pandas').content[0]
        results = pd.DataFrame(columns=['driverCode', 'constructorId', 'position'])
        results['driverCode'] = qualiRes['driverCode']
        results['constructorId'] = qualiRes['constructorId']
        results['position'] = qualiRes['position']
        quali_df = pd.concat([quali_df, results])
        new_round -= 1

    quali_df = quali_df.reset_index()
    quali_df = quali_df.drop(columns='index')
    return quali_df


def prev_team_id(season, teamId):
    if season == 2019:
        if teamId == "racing_point":
            return "force_india"
    elif season == 2020:
        if teamId == "alphatauri":
            return "toro_rosso"
    elif season == 2021:
        if teamId == "alpine":
            return "renault"
        elif teamId == "aston_martin":
            return "racing_point"
    elif season == 2024:
        if teamId == "rb":
            return "alphatauri"
        elif teamId == "sauber":
            return "alfa"
    else:
        return None


def f1_era(season):
    if 2017 <= season <= 2021:
        return 'WideAero'
    else:
        return 'GroundEffect'


def get_driver_seasons(season):
    if season == 2018:
        data = {'driverCode': ['ALO', 'BOT', 'ERI', 'GAS', 'GRO', 'HAM', 'HAR', 'HUL', 'LEC', 'MAG', 'OCO', 'PER', 'RAI', 'RIC', 'SAI', 'SIR', 'STR', 'VAN', 'VER', 'VET'],
                'seasonsCount': [16, 5, 4, 1, 7, 11, 1, 7, 0, 3, 2, 7, 15, 7, 3, 0, 1, 1, 3, 11]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2019:
        data = {'driverCode': ['HAM', 'BOT', 'VER', 'LEC', 'VET', 'SAI', 'GAS', 'ALB', 'RIC', 'PER', 'NOR', 'RAI', 'KVY', 'HUL', 'STR', 'MAG', 'GIO', 'GRO', 'KUB',
                               'RUS'],
                'seasonsCount': [12, 6, 4, 1, 12, 4, 2, 0, 8, 8, 0, 16, 4, 8, 2, 4, 1, 8, 5, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2020:
        data = {'driverCode': ['HAM', 'BOT', 'VER', 'PER', 'RIC', 'SAI', 'ALB', 'LEC', 'NOR', 'GAS', 'STR', 'OCO', 'VET', 'KVY', 'HUL', 'RAI', 'GIO', 'RUS', 'GRO', 'MAG', 'LAT', 'AIT', 'FIT'],
                'seasonsCount': [13, 7, 5, 9, 9, 5, 1, 2, 1, 3, 3, 4, 13, 5, 9, 17, 2, 1, 9, 5, 0, 0, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2021:
        data = {'driverCode': ['VER', 'HAM', 'BOT', 'PER', 'SAI', 'NOR', 'LEC', 'RIC', 'GAS', 'ALO', 'OCO', 'VET', 'STR', 'TSU', 'RUS', 'RAI', 'LAT', 'GIO', 'MSC', 'KUB', 'MAZ'],
                'seasonsCount': [6,  14,  8,  10,  6,  2,  3,  10,  4,  17,  5,  14,  4,  0,  2,  18,  1,  3,  0,  7,  0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2022:
        data = {'driverCode': ['VER', 'LEC', 'PER', 'RUS', 'SAI', 'HAM', 'NOR', 'OCO', 'ALO', 'BOT', 'RIC', 'VET', 'MAG', 'GAS', 'STR', 'MSC', 'TSU', 'ZHO', 'ALB', 'LAT', 'DEV', 'HUL'],
                'seasonsCount': [7, 4, 11, 3, 7, 15, 3, 6, 18, 9, 11, 15, 7, 5, 5, 1, 1, 0, 2, 2, 0, 10]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2023:
        data = {'driverCode': ['VER','PER','HAM','ALO','LEC','NOR','SAI','RUS','PIA','STR','GAS','OCO','ALB','TSU','BOT','HUL','RIC','ZHO','MAG','LAW','SAR','DEV'],
                'seasonsCount': [8, 12, 16, 19, 5, 4, 8, 4, 0, 6, 6, 7, 4, 2, 10, 10, 12, 1, 8, 0, 0, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2024:
        data = {'driverCode': ['VER','NOR','LEC','PIA','SAI','RUS','HAM','PER','ALO','GAS','HUL','TSU','STR','OCO','MAG','ALB','RIC','BEA','COL','ZHO','LAW','BOT','SAR','DOO'],
                'seasonsCount': [9, 5, 6, 1, 9, 5, 17, 13, 20, 7, 13, 3, 7, 8, 9, 4, 13, 0, 0, 2, 1, 11, 1, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2025:
        data = {'driverCode': ['PIA','NOR','VER','RUS','LEC','ANT','HAM','ALB','OCO','STR','HUL','GAS','BEA','HAD','SAI','TSU','ALO','LAW','DOO','BOR'],
                'seasonsCount': [2,6,10,6,7,5,18,6,9,8,12,8,0,0,10,4,21,1,0,0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf


def get_engine_seasons(season):
    if season == 2018:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Renault", "Haas F1 Team", "McLaren", "Force India", "Racing Point", "Toro Rosso", "Sauber", "Williams"],
                'engine': ["Mercedes", "Ferrari", "Renault", "Renault", "Ferrari", "Renault", "Mercedes", "Mercedes", "Honda", "Ferrari", "Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2019:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Renault","Haas F1 Team","McLaren","Racing Point","Toro Rosso","Alfa Romeo Racing","Williams"],
                'engine': ["Mercedes","Ferrari","Honda","Renault","Ferrari","Renault","Mercedes","Honda","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2020:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Renault","Haas F1 Team","McLaren","Racing Point","AlphaTauri","Alfa Romeo Racing","Williams"],
                'engine': ["Mercedes","Ferrari","Honda","Renault","Ferrari","Renault","Mercedes","Honda","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2021:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Alpine","Haas F1 Team","McLaren","Aston Martin","AlphaTauri","Alfa Romeo Racing","Williams"],
                'engine': ["Mercedes","Ferrari","Honda","Renault","Ferrari","Mercedes","Mercedes","Honda","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2022:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Alpine", "Haas F1 Team", "McLaren", "Aston Martin", "AlphaTauri", "Alfa Romeo", "Williams"],
                'engine': ["Mercedes","Ferrari","Honda RBPT","Renault","Ferrari","Mercedes","Mercedes","Honda RBPT","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2023:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Alpine","Haas F1 Team","McLaren","Aston Martin","AlphaTauri","Alfa Romeo","Williams"],
                'engine': ["Mercedes","Ferrari","Honda RBPT","Renault","Ferrari","Mercedes","Mercedes","Honda RBPT","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2024:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Alpine","Haas F1 Team","McLaren","Aston Martin","RB","Kick Sauber","Williams"],
                'engine': ["Mercedes","Ferrari","Honda RBPT","Renault","Ferrari","Mercedes","Mercedes","Honda RBPT","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2025:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Alpine", "Haas F1 Team", "McLaren", "Aston Martin", "Racing Bulls", "Kick Sauber", "Williams"],
                'engine': ["Mercedes","Ferrari","Honda RBPT","Renault","Ferrari","Mercedes","Mercedes","Honda RBPT","Ferrari","Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf


if __name__ == '__main__':
    year = 2025
    airTemp = 27.2
    pressure = 1014.5
    humidity = 69.0
    rainfall = 0
    circuitId = 'miami'
    drivers_data = {
        'Driver': ['VER', 'NOR', 'ANT', 'PIA', 'RUS', 'SAI', 'ALB', 'LEC', 'OCO', 'TSU',
                   'HAD', 'HAM', 'BOR', 'COL', 'LAW', 'HUL', 'ALO', 'GAS', 'STR', 'BEA']}
    teams_data = {
        'Team': ['Red Bull Racing', 'McLaren', 'Mercedes', 'McLaren', 'Mercedes', 'Williams', 'Williams',
                 'Ferrari', 'Haas F1 Team', 'Red Bull Racing', 'Racing Bulls', 'Ferrari', 'Kick Sauber',
                 'Alpine', 'Racing Bulls', 'Kick Sauber', 'Aston Martin', 'Alpine', 'Aston Martin', 'Haas F1 Team']
    }

    df_driver = pd.DataFrame(drivers_data)
    df_teams = pd.DataFrame(teams_data)
    df = pd.concat([df_driver, df_teams], axis=1)
    print(df)
    test_data(year, airTemp, pressure, humidity, rainfall, "SQ", 6, df, 19, 5.412, 1)
