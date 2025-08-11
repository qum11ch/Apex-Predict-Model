from main import load_single_file
import numpy as np
import fastf1
from fastf1.ergast import Ergast
import pandas as pd
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ergast = Ergast(result_type='pandas', auto_cast=True)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def add_quali_data():
    gp_list = ['Australia', 'Shanghai', 'Suzuka', 'Bahrain', 'Jeddah', 'Miami', 'Imola', 'Monaco', 'Spain', 'Canada',
               'Austria', 'Britain', 'Belgium', 'Hungary', 'Netherland', 'Monza', 'Baku', 'Singapore', 'Austin',
               'Mexico', 'Brazil', 'Las Vegas', 'Lusail', 'Abu Dhabi']
    gp_list_ended = ['Australia', 'Shanghai', 'Suzuka', 'Bahrain', 'Jeddah', 'Miami']

    for gp in gp_list:
        if gp in gp_list_ended:
            last_year = 2026
        else:
            last_year = 2025
        for year in range(2018, last_year):
            event1 = fastf1.get_session(year, gp, 1)
            if (event1.event['Location'] == gp) or (event1.event['Country'] == gp):
                gpRound = event1.event['RoundNumber']
                race = ergast.get_race_schedule(season=year, round=gpRound)

                currentSeasonQuali = None
                if year > 2024:
                    currentSeasonQuali = get_current_season_quali(year, gpRound)

                driver = get_driver_standings(year, gpRound)
                constructor = get_constructor_standings(year, gpRound)

                if (year == 2020) & (gp == 'Imola'):
                    data = {
                        'name': ['None']
                    }

                    event3 = pd.DataFrame(data)
                else:
                    event3 = fastf1.get_session(year, gp, 3)

                event4 = fastf1.get_session(year, gp, 4)

                event4.load(laps=True, weather=True)
                if year == 2018:
                    event4_compounds = event4.laps.pick_compounds(
                        ['HYPERSOFT', 'ULTRASOFT', 'SUPERSOFT', 'SOFT']).pick_quicklaps()
                else:
                    event4_compounds = event4.laps.pick_compounds(['SOFT', 'MEDIUM']).pick_quicklaps()

                if (year == 2021) | (year == 2022) | (year == 2023):
                    if event4.name != "Qualifying":
                        event2 = fastf1.get_session(year, gp, 2)

                        event2.load(laps=True, weather=True)

                        event2_compounds = event2.laps.pick_compounds(['SOFT', 'MEDIUM']).pick_quicklaps()

                        if event2_compounds.size != 0:
                            q1, q2, q3 = event2_compounds.split_qualifying_sessions()
                            if q1 is not None:
                                q1['Event'] = "Q1"
                            if q2 is not None:
                                q2['Event'] = "Q2"
                            if q3 is not None:
                                q3['Event'] = "Q3"
                            event2_total = pd.concat([q1, q2, q3], ignore_index=True)

                            event2_weather = event2.laps.get_weather_data()
                            event2_weather = event2_weather.reset_index(drop=True)

                            event2_data = pd.concat(
                                [event2_total, event2_weather.loc[:, ~(event2_weather.columns == 'Time')]],
                                axis=1)

                            event2_data = event2_data.dropna(subset='Driver')

                            event2_data = add_season_stats(event2_data, driver, constructor, gpRound, year,
                                                           currentSeasonQuali, event2)
                        else:
                            event2_data = None

                        if year == 2023:
                            event3.load(laps=True, weather=True)

                            event3_compounds = event3.laps.pick_compounds(['SOFT', 'MEDIUM']).pick_quicklaps()

                            if event3_compounds.size != 0:
                                sq1, sq2, sq3 = event3_compounds.split_qualifying_sessions()
                                if sq1 is not None:
                                    sq1['Event'] = "SQ1"
                                if sq2 is not None:
                                    sq2['Event'] = "SQ2"
                                if sq3 is not None:
                                    sq3['Event'] = "SQ3"
                                event3_total = pd.concat([sq1, sq2, sq3], ignore_index=True)

                                event3_weather = event3.laps.get_weather_data()
                                event3_weather = event3_weather.reset_index(drop=True)

                                event3_data = pd.concat(
                                    [event3_total, event3_weather.loc[:, ~(event3_weather.columns == 'Time')]],
                                    axis=1)

                                event3_data = event3_data.dropna(subset='Driver')

                                event3_data = add_season_stats(event3_data, driver, constructor, gpRound, year,
                                                               currentSeasonQuali, event3)
                            else:
                                event3_data = None

                            if (event2_data is not None) | (event3_data is not None):
                                frames = [event2_data, event3_data]
                            else:
                                frames = None
                        else:
                            if event2_data is not None:
                                frames = [event2_data]
                            else:
                                frames = None
                    else:
                        if event4_compounds.size != 0:
                            q1, q2, q3 = event4_compounds.split_qualifying_sessions()
                            if q1 is not None:
                                q1['Event'] = "Q1"
                            if q2 is not None:
                                q2['Event'] = "Q2"
                            if q3 is not None:
                                q3['Event'] = "Q3"

                            event4_total = pd.concat([q1, q2, q3], ignore_index=True)

                            event4_weather = event4.laps.get_weather_data()
                            event4_weather = event4_weather.reset_index(drop=True)

                            event4_data = pd.concat(
                                [event4_total, event4_weather.loc[:, ~(event4_weather.columns == 'Time')]],
                                axis=1)
                            event4_data = event4_data.dropna(subset='Driver')
                            event4_data = add_season_stats(event4_data, driver, constructor, gpRound, year,
                                                           currentSeasonQuali,
                                                           event4)
                        else:
                            event4_data = None

                        if event4_data is not None:
                            frames = [event4_data]
                        else:
                            frames = None
                else:
                    if event4_compounds.size != 0:
                        q1, q2, q3 = event4_compounds.split_qualifying_sessions()
                        if q1 is not None:
                            q1['Event'] = "Q1"
                        if q2 is not None:
                            q2['Event'] = "Q2"
                        if q3 is not None:
                            q3['Event'] = "Q3"
                        event4_total = pd.concat([q1, q2, q3], ignore_index=True)

                        event4_weather = event4.laps.get_weather_data()
                        event4_weather = event4_weather.reset_index(drop=True)

                        event4_data = pd.concat(
                            [event4_total, event4_weather.loc[:, ~(event4_weather.columns == 'Time')]],
                            axis=1)
                        event4_data = event4_data.dropna(subset='Driver')
                        event4_data = add_season_stats(event4_data, driver, constructor, gpRound, year,
                                                       currentSeasonQuali,
                                                       event4)
                    else:
                        event4_data = None

                    if (year == 2020) & (gp == 'Imola'):
                        reg = "Practice 3"
                    else:
                        reg = event3.name

                    if reg != "Practice 3":
                        event2 = fastf1.get_session(year, gp, 2)

                        event2.load(laps=True, weather=True)

                        event2_compounds = event2.laps.pick_compounds(['SOFT', 'MEDIUM']).pick_quicklaps()

                        if event2_compounds.size != 0:
                            sq1, sq2, sq3 = event2_compounds.split_qualifying_sessions()
                            if sq1 is not None:
                                sq1['Event'] = "SQ1"
                            if sq2 is not None:
                                sq2['Event'] = "SQ2"
                            if sq3 is not None:
                                sq3['Event'] = "SQ3"

                            event2_total = pd.concat([sq1, sq2, sq3], ignore_index=True)

                            event2_weather = event2.laps.get_weather_data()
                            event2_weather = event2_weather.reset_index(drop=True)

                            event2_data = pd.concat(
                                [event2_total, event2_weather.loc[:, ~(event2_weather.columns == 'Time')]],
                                axis=1)

                            event2_data = event2_data.dropna(subset='Driver')

                            event2_data = add_season_stats(event2_data, driver, constructor, gpRound, year,
                                                           currentSeasonQuali,
                                                           event2)
                        else:
                            event2_data = None

                        if (event2_data is not None) | (event4_data is not None):
                            frames = [event2_data, event4_data]
                        else:
                            frames = None
                    else:
                        if event4_data is not None:
                            frames = [event4_data]
                        else:
                            frames = None

                if frames is not None:
                    joinedData = pd.concat(frames)
                    joinedData = joinedData.reset_index()

                    joinedData = joinedData.query('Deleted == False & TrackStatus == "1"')
                    joinedData = joinedData[['Driver', 'Team', 'LapTime', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
                                             'TrackTemp', 'DriverPoints', 'TeamPoints', 'DriverAvgQualiPos',
                                             'TeamAvgQualiPos', 'Event']]
                    circuitsLenDf = get_circuits_len()

                    joinedData['LapTime'] = joinedData['LapTime'].dt.total_seconds()
                    joinedData['CircuitId'] = None
                    joinedData['CircuitId'].values[:] = race.circuitId.values[0]
                    joinedData['Year'] = None
                    joinedData['Year'].values[:] = year
                    joinedData['IsStreetCircuit'] = None
                    joinedData['IsStreetCircuit'].values[:] = is_street_circuit(race.circuitId.values[0])
                    joinedData['F1Era'] = None
                    joinedData['F1Era'].values[:] = f1_era(year)
                    joinedData['CircuitCorners'] = None
                    joinedData['CircuitCorners'].values[:] = event4.get_circuit_info().corners['Number'].values[-1]
                    joinedData['CircuitLength'] = None
                    joinedData['CircuitLength'].values[:] = \
                        circuitsLenDf[circuitsLenDf['Circuit'] == race.circuitId.values[0]][
                            'LapLength_km'].item()
                    joinedData['TrackAirDiff'] = None
                    joinedData['IsHighHumidity'] = None
                    joinedData['DriverQualiPace'] = None
                    joinedData['DriverSeasons'] = None
                    joinedData['TeamPointsContribution'] = None
                    joinedData['CarEngine'] = None

                    driversSeasonDf = get_driver_seasons(season=year)
                    engineSeasonDf = get_engine_seasons(season=year)

                    for i, j in joinedData.iterrows():
                        airTemp = j['AirTemp']
                        trackTemp = j['TrackTemp']
                        humidity = j['Humidity']
                        driverAvgQuali = j['DriverAvgQualiPos']
                        teamAvgQuali = j['TeamAvgQualiPos']
                        driverCode = j['Driver']
                        driverPoints = j['DriverPoints']
                        teamPoints = j['TeamPoints']
                        teamName = j['Team']

                        trackAirDiff = round((trackTemp - airTemp), 2)
                        isHighHumidity = False
                        if humidity > 70.0:
                            isHighHumidity = True

                        if driverAvgQuali != 0.0:
                            driverQualiPace = round((teamAvgQuali - driverAvgQuali), 2)
                        else:
                            driverQualiPace = 0.0

                        if teamPoints != 0.0:
                            teamPointsContribution = round((driverPoints / teamPoints), 2)
                        else:
                            teamPointsContribution = 0.5

                        joinedData.at[i, 'TeamPointsContribution'] = teamPointsContribution
                        joinedData.at[i, 'TrackAirDiff'] = trackAirDiff
                        joinedData.at[i, 'IsHighHumidity'] = isHighHumidity
                        joinedData.at[i, 'DriverQualiPace'] = driverQualiPace

                        seasonsCount = driversSeasonDf[driversSeasonDf['driverCode'] == driverCode][
                            'seasonsCount'].values
                        carEngine = engineSeasonDf[engineSeasonDf['teamName'] == teamName]['engine'].values

                        if carEngine.size != 0:
                            joinedData.at[i, 'CarEngine'] = carEngine.item()
                        else:
                            joinedData.at[i, 'CarEngine'] = None

                        if seasonsCount.size != 0:
                            joinedData.at[i, 'DriverSeasons'] = seasonsCount.item()
                        else:
                            k = year - 1
                            while k > 2017:
                                prevDriversSeasonDf = get_driver_seasons(season=k)
                                newSeasonsCount = prevDriversSeasonDf[prevDriversSeasonDf['driverCode'] == driverCode][
                                    'seasonsCount'].values
                                if newSeasonsCount.size != 0:
                                    joinedData.at[i, 'DriverSeasons'] = newSeasonsCount.item()
                                    break
                                else:
                                    k -= 1
                            if k == 2017:
                                joinedData.at[i, 'DriverSeasons'] = 0.0

                    finalCSV = pd.concat([df, joinedData])
                    df = finalCSV

    name = "dataset/QualiData.csv"
    df.to_csv(name, index=False)


def add_season_stats(df, driver, constructor, raceRound, year, currentSeasonQualis, event):
    drivers_df = load_single_file("C:/Users/PC/Desktop/f1_py/ergast/drivers.csv")
    qualifying_df = load_single_file("C:/Users/PC/Desktop/f1_py/ergast/qualifying.csv")
    races_df = load_single_file("C:/Users/PC/Desktop/f1_py/ergast/races.csv")
    constructors_df = load_single_file("C:/Users/PC/Desktop/f1_py/ergast/constructors.csv")

    df[['DriverPoints', 'TeamPoints', 'DriverAvgQualiPos', 'TeamAvgQualiPos']] = None

    for i, j in df.iterrows():
        driverCode = j['Driver']
        driverDf = driver[driver['Driver'] == driverCode]

        teamName = j['Team']

        driverId = driverDf['DriverId'].astype('string').values
        driverId_df = drivers_df[drivers_df['driverRef'].isin(driverId)]['driverId'].values

        if raceRound == 1:
            currentSeason = ergast.get_driver_standings(season=year, round=raceRound).content[0]
            teamId = currentSeason[currentSeason['driverCode'] == driverCode]['constructorIds'].values[0]
            teamId = np.stack(teamId, axis=0)
        else:
            teamId = driver[driver['Driver'] == driverCode]['TeamId'].values

        if teamId.size == 0:
            if (year == 2020) & (raceRound == 4) & (driverCode == "HUL"):
                driverTeam = "Racing Point"
            else:
                driverTeam = df[df['Driver'] == driverCode]['Team'].values[0]

            event_laps = event.laps
            otherDrivers = event_laps[event_laps['Team'] == driverTeam]
            otherDriverCode = otherDrivers[otherDrivers['Driver'] != driverCode]['Driver'].values[0]

            teamId = driver[driver['Driver'] == otherDriverCode]['TeamId'].values

            if teamId.size == 0:
                if teamName == "Red Bull Racing":
                    teamId = np.array(['red_bull'])

        teamDf = constructor[constructor['Constructor'].isin(teamId)]

        prevConstructorStandings = get_constructor_standings(year, 1)

        prevTeamDf = prevConstructorStandings[prevConstructorStandings['Constructor'].isin(teamId)]

        if prevTeamDf.size == 0:
            prev_teamId = prev_team_id(year, teamId[0])
            if prev_teamId is not None:
                prevTeamDf = prevConstructorStandings[prevConstructorStandings['Constructor'] == prev_teamId]

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
            df.at[i, 'DriverPoints'] = driverDf['Points'].values.astype(float)

        else:
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

                teamQualis = currentSeasonQualis.loc[currentSeasonQualis['constructorId'].isin(teamId)][
                    'position'].values
                team_quali_count = teamQualis.size

                if driver_quali_count != 0:
                    df.at[i, 'DriverAvgQualiPos'] = round(driverQualis.mean(), 2)
                else:
                    df.at[i, 'DriverAvgQualiPos'] = 0.0

                if team_quali_count != 0:
                    df.at[i, 'TeamAvgQualiPos'] = round(teamQualis.mean(), 2)
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


def is_street_circuit(circuitId):
    street_circuits_list = ['albert_park', 'baku', 'monaco', 'villeneuve', 'jeddah', 'vegas', 'miami', 'marina_bay']
    if circuitId in street_circuits_list:
        return True
    else:
        return False


def get_circuits_len():
    data = {
        'Circuit': [
            'albert_park', 'shanghai', 'suzuka', 'bahrain', 'jeddah', 'miami', 'imola', 'monaco', 'catalunya',
            'villeneuve', 'red_bull_ring', 'silverstone', 'spa', 'hungaroring', 'zandvoort', 'monza', 'baku',
            'marina_bay', 'americas', 'rodriguez', 'interlagos', 'vegas', 'losail', 'yas_marina'],
        'LapLength_km': [5.278, 5.451, 5.807, 5.412, 6.174, 5.412, 4.909, 3.337, 4.657, 4.361, 4.318, 5.891, 7.004,
                         4.381, 4.259, 5.793, 6.003, 4.94, 5.513, 4.304, 4.309, 6.201, 5.419, 5.281]
    }
    df = pd.DataFrame(data)
    return df


def get_driver_seasons(season):
    if season == 2018:
        data = {'driverCode': ['ALO', 'BOT', 'ERI', 'GAS', 'GRO', 'HAM', 'HAR', 'HUL', 'LEC', 'MAG', 'OCO', 'PER', 'RAI',
                               'RIC', 'SAI', 'SIR', 'STR', 'VAN', 'VER', 'VET'],
                'seasonsCount': [16, 5, 4, 1, 7, 11, 1, 7, 0, 3, 2, 7, 15, 7, 3, 0, 1, 1, 3, 11]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2019:
        data = {'driverCode': ['HAM', 'BOT', 'VER', 'LEC', 'VET', 'SAI', 'GAS', 'ALB', 'RIC', 'PER', 'NOR', 'RAI',
                               'KVY', 'HUL', 'STR', 'MAG', 'GIO', 'GRO', 'KUB', 'RUS'],
                'seasonsCount': [12, 6, 4, 1, 12, 4, 2, 0, 8, 8, 0, 16, 4, 8, 2, 4, 1, 8, 5, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2020:
        data = {'driverCode': ['HAM', 'BOT', 'VER', 'PER', 'RIC', 'SAI', 'ALB', 'LEC', 'NOR', 'GAS', 'STR', 'OCO', 'VET',
                               'KVY', 'HUL', 'RAI', 'GIO', 'RUS', 'GRO', 'MAG', 'LAT', 'AIT', 'FIT'],
                'seasonsCount': [13, 7, 5, 9, 9, 5, 1, 2, 1, 3, 3, 4, 13, 5, 9, 17, 2, 1, 9, 5, 0, 0, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2021:
        data = {'driverCode': ['VER', 'HAM', 'BOT', 'PER', 'SAI', 'NOR', 'LEC', 'RIC', 'GAS', 'ALO', 'OCO', 'VET', 'STR',
                               'TSU', 'RUS', 'RAI', 'LAT', 'GIO', 'MSC', 'KUB', 'MAZ'],
                'seasonsCount': [6, 14, 8, 10, 6, 2, 3, 10, 4, 17, 5, 14, 4, 0, 2, 18, 1, 3, 0, 7, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2022:
        data = {'driverCode': ['VER', 'LEC', 'PER', 'RUS', 'SAI', 'HAM', 'NOR', 'OCO', 'ALO', 'BOT', 'RIC', 'VET', 'MAG',
                               'GAS', 'STR', 'MSC', 'TSU', 'ZHO', 'ALB', 'LAT', 'DEV', 'HUL'],
                'seasonsCount': [7, 4, 11, 3, 7, 15, 3, 6, 18, 9, 11, 15, 7, 5, 5, 1, 1, 0, 2, 2, 0, 10]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2023:
        data = {'driverCode': ['VER', 'PER', 'HAM', 'ALO', 'LEC', 'NOR', 'SAI', 'RUS', 'PIA', 'STR', 'GAS', 'OCO', 'ALB',
                               'TSU', 'BOT', 'HUL', 'RIC', 'ZHO', 'MAG', 'LAW', 'SAR', 'DEV'],
                'seasonsCount': [8, 12, 16, 19, 5, 4, 8, 4, 0, 6, 6, 7, 4, 2, 10, 10, 12, 1, 8, 0, 0, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2024:
        data = {'driverCode': ['VER', 'NOR', 'LEC', 'PIA', 'SAI', 'RUS', 'HAM', 'PER', 'ALO', 'GAS', 'HUL', 'TSU', 'STR',
                               'OCO', 'MAG', 'ALB', 'RIC', 'BEA', 'COL', 'ZHO', 'LAW', 'BOT', 'SAR', 'DOO'],
                'seasonsCount': [9, 5, 6, 1, 9, 5, 17, 13, 20, 7, 13, 3, 7, 8, 9, 4, 13, 0, 0, 2, 1, 11, 1, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2025:
        data = {'driverCode': ['PIA', 'NOR', 'VER', 'RUS', 'LEC', 'ANT', 'HAM', 'ALB', 'OCO', 'STR', 'HUL', 'GAS', 'BEA',
                               'HAD', 'SAI', 'TSU', 'ALO', 'LAW', 'DOO', 'BOR'],
                'seasonsCount': [2, 6, 10, 6, 7, 5, 18, 6, 9, 8, 12, 8, 0, 0, 10, 4, 21, 1, 0, 0]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf


def get_engine_seasons(season):
    if season == 2018:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Renault", "Haas F1 Team", "McLaren", "Force India",
                             "Racing Point", "Toro Rosso", "Sauber", "Williams"],
                'engine': ["Mercedes", "Ferrari", "Renault", "Renault", "Ferrari", "Renault", "Mercedes", "Mercedes",
                           "Honda", "Ferrari", "Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2019:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Renault", "Haas F1 Team", "McLaren",
                             "Racing Point", "Toro Rosso", "Alfa Romeo Racing", "Williams"],
                'engine': ["Mercedes", "Ferrari", "Honda", "Renault", "Ferrari", "Renault", "Mercedes", "Honda",
                           "Ferrari", "Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2020:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Renault", "Haas F1 Team", "McLaren",
                             "Racing Point", "AlphaTauri", "Alfa Romeo Racing", "Williams"],
                'engine': ["Mercedes", "Ferrari", "Honda", "Renault", "Ferrari", "Renault", "Mercedes",
                           "Honda", "Ferrari", "Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2021:
        data = {'teamName': ["Mercedes", "Ferrari", "Red Bull Racing", "Alpine", "Haas F1 Team", "McLaren",
                             "Aston Martin", "AlphaTauri", "Alfa Romeo Racing", "Williams"],
                'engine': ["Mercedes","Ferrari","Honda","Renault","Ferrari","Mercedes","Mercedes","Honda","Ferrari",
                           "Mercedes"]}
        seasonsDf = pd.DataFrame(data=data)
        return seasonsDf
    elif season == 2022:
        data = {'teamName': ["Mercedes","Ferrari","Red Bull Racing","Alpine","Haas F1 Team","McLaren","Aston Martin", "AlphaTauri", "Alfa Romeo", "Williams"],
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


def dataset_main():
    add_quali_data()


if __name__ == '__main__':
    dataset_main()

