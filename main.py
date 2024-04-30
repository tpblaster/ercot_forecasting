import datetime
from io import BytesIO
from math import sqrt
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from numpy import concatenate
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, USMemorialDay, USLaborDay, USThanksgivingDay, \
    sunday_to_monday

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.arima.model import ARIMA

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

ERCOT_API_KEY = "6ce5fd416deb47218c1da1d1678e1395"
ERCOT_PASSWORD = "XcZ8bg9t2HrbAXh"


def get_id_token():
    url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token?"
    params = {
        "grant_type": "password",
        "username": "promalcolm@gmail.com",
        "password": ERCOT_PASSWORD,
        "response_type": "id_token",
        "scope": "openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access",
        "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70"
    }

    url_params = "&".join([f"{p}={v}" for p, v in params.items()])
    r = requests.post(url + url_params)
    return r.json()["access_token"]


def get_weather_zone_actuals(bearer):
    url = "https://api.ercot.com/api/public-reports/np6-345-cd/act_sys_load_by_wzn"
    headers = {
        "authorization": "Bearer " + bearer,
        "Ocp-Apim-Subscription-Key": ERCOT_API_KEY
    }
    params = {
        "operatingDayFrom": "2022-06-01",
        "operatingDayTo": "2022-06-07",
        "page": 1,
        "size": 100,
        # "coastFrom": 1,
        # "coastTo": 2,
    }
    r = requests.get(url, headers=headers, params=params)
    return r.json()


def ercot_to_datetime(date_str):
    if date_str[11:13] != '24':
        return pd.to_datetime(date_str[0:16], format='%m/%d/%Y %H:%M')

    date_str = date_str[0:11] + '00' + date_str[13:16]
    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M') + datetime.timedelta(days=1)


def get_eroct_load_data_annual(zip_url: str):
    url = requests.get(zip_url)
    zip_file = ZipFile(BytesIO(url.content))

    load_data = pd.read_excel(zip_file.open(zip_file.namelist()[0]))

    # fixing issue with ERCOT data
    if zip_url[-8:] == "2022.zip":
        load_data.loc[8016, "Hour Ending"] = "12/01/2022 01:00"

    load_data[["DATE", "HOUR ENDING", "DST"]] = load_data.iloc[:, 0].str.split(" ", expand=True)
    load_data.rename(columns={load_data.columns[0]: "DATETIME"}, inplace=True)
    load_data["DST"] = load_data["DST"].astype('bool').astype('int')
    load_data["DATE"] = pd.to_datetime(load_data["DATE"])
    load_data["HOUR ENDING"] = load_data["HOUR ENDING"].str[0:2].astype('int')
    load_data["DATETIME"] = load_data["DATETIME"].apply(ercot_to_datetime)
    return load_data


def get_ercot_load(load_urls):
    total_data = []
    for url in load_urls:
        total_data.append(get_eroct_load_data_annual(url))
    return pd.concat(total_data, ignore_index=True)


class NERCHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=sunday_to_monday),
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=sunday_to_monday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=sunday_to_monday)
    ]


def build_ercot_load():
    cal = NERCHolidayCalendar()
    urls = [
        "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
        "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
        "https://www.ercot.com/files/docs/2021/11/12/Native_Load_2021.zip",
        "https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip",
        "https://www.ercot.com/files/docs/2020/01/09/Native_Load_2019.zip",
        "https://www.ercot.com/files/docs/2019/01/07/native_load_2018.zip",
    ]
    # https://www.cmegroup.com/content/dam/cmegroup/rulebook/NYMEX/2/283.pdf
    ercot_holidays = cal.holidays(datetime.datetime(2018, 1, 1), datetime.datetime(2023, 12, 31))
    load = get_ercot_load(urls)
    load.sort_values(by=["DATE", "HOUR ENDING", "DST"], inplace=True)
    load["HOLIDAY"] = load["DATE"].isin(ercot_holidays).astype('int')
    load["DAYOFWEEK"] = load["DATE"].dt.weekday
    load["MONTH"] = load["DATE"].dt.month
    load["WEEKEND"] = (load["DAYOFWEEK"] >= 5).astype('int')
    load["PEAK"] = ((~load["HOLIDAY"]) & (~load["WEEKEND"]) & (load["HOUR ENDING"] > 6) & (
            load["HOUR ENDING"] < 23)).astype('int')
    load.reset_index(drop=True, inplace=True)
    return load


def prepare_training_data(lag=3):
    ercot = build_ercot_load()
    # consider min max scaling
    ercot = ercot[['ERCOT', 'HOUR ENDING', 'DAYOFWEEK', 'MONTH']]
    lagged = [ercot]

    for i in range(1, lag + 1):
        print(i)
        lagged.append(ercot.shift(lag).add_suffix(f"(t-{lag})"))

    ercot = pd.concat(lagged, ignore_index=True, axis=1)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = pd.DataFrame(scaler.fit_transform(ercot.values.astype('float32')))
    scaled[len(scaled.columns)] = scaled[0].shift(-1)
    scaled.dropna(inplace=True)

    values = scaled.values

    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    plt.plot(inv_y[:24 * 14], label="actual")
    plt.plot(inv_yhat[:24 * 14], label="predicted")
    plt.legend()
    plt.title("LSTM Model with 3 Lagged Inputs")
    plt.xlabel("Day")
    plt.ylabel("Load(Megawatts)")
    plt.savefig("lstm_test.png")
    print('Test RMSE: %.3f' % rmse)


def zone_decomposition_forecast(zone):
    ercot = build_ercot_load()
    train = ercot.iloc[:(len(ercot) - 365 * 24)]
    test = ercot.iloc[(len(ercot) - 365 * 24):]

    mstl = MSTL(train.set_index("DATETIME")[zone],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    year_offsets = res.seasonal["seasonal_8760"].head(8760)
    year_offsets.to_csv("snaive-model.csv")

    # final model selected by AIC
    mod = ARIMA(res.seasonal["seasonal_24"], order=(3, 1, 3), seasonal_order=(3, 1, 2, 24))
    day_model = mod.fit(method='innovations_mle', low_memory=True, cov_type='none')
    day_model.save("day_ercot.pkl")

    mod = ARIMA(res.seasonal["seasonal_168"], order=(1, 1, 1), seasonal_order=(2, 1, 2, 24 * 7))
    week_model = mod.fit(method='innovations_mle', low_memory=True, cov_type='none')
    week_model.save("week_ercot.pkl")

    # day_model = auto_arima(y=res.seasonal["seasonal_24"], seasonal=True, m=24, trace=True, stepwise=True, d=1, D=1,
    #                        fit_args={"method": 'innovations_mle', "low_memory": True, "cov_type": 'none'})
    # day_model.save("day_auto_ercot.pkl")


def pred_svr_model(input_frame, timesteps, pred_count):
    scaler = MinMaxScaler()
    train = scaler.fit_transform(input_frame)
    train_data_timesteps = np.array(
        [[j for j in train[i:i + timesteps]] for i in range(0, len(train) - timesteps + 1)])[:, :, 0]

    x_train, y_train = train_data_timesteps[:, :timesteps - 1], train_data_timesteps[:, [timesteps - 1]]

    model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
    model.fit(x_train, y_train[:, 0])

    next_in = x_train[-1]

    preds = []
    for _ in range(pred_count):
        pred = model.predict(next_in.reshape(1, -1))[0]
        preds.append(pred)
        next_in = np.roll(next_in, -1)
        next_in[-1] = pred

    scaled = scaler.inverse_transform(np.array(preds).reshape(1, -1))[0]
    return pd.Series(scaled)


def create_svr(input_frame, timesteps=25):
    train = input_frame.iloc[:(len(input_frame) - 365 * 24)]
    test = input_frame.iloc[(len(input_frame) - 365 * 24):]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    train_data_timesteps = np.array(
        [[j for j in train[i:i + timesteps]] for i in range(0, len(train) - timesteps + 1)])[:, :, 0]

    test_data_timesteps = np.array([[j for j in test[i:i + timesteps]] for i in range(0, len(test) - timesteps + 1)])[:,
                          :, 0]

    x_train, y_train = train_data_timesteps[:, :timesteps - 1], train_data_timesteps[:, [timesteps - 1]]
    x_test, y_test = test_data_timesteps[:, :timesteps - 1], test_data_timesteps[:, [timesteps - 1]]

    model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
    model.fit(x_train, y_train[:, 0])

    y_train_pred = model.predict(x_train).reshape(-1, 1)
    y_test_pred = model.predict(x_test).reshape(-1, 1)

    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)

    mape_train = mean_absolute_percentage_error(y_train_pred, y_train) * 100
    mape_test = mean_absolute_percentage_error(y_test_pred, y_test) * 100
    return mape_train, mape_test


def test_svr_params():
    ercot = build_ercot_load()

    mstl = MSTL(ercot["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    hist = {}
    for x in range(5, 49):
        print(x)
        hist[x] = create_svr(pd.DataFrame(res.trend + res.resid, columns=["ERCOT"]), x)

    pd.DataFrame(hist).T.to_csv("timestep_analysis.csv", index_label="Timesteps")


def test_svr_preds():
    ercot = build_ercot_load()

    mstl = MSTL(ercot["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    trend_error = pd.DataFrame(res.trend + res.resid, columns=["ERCOT"])

    for day in range(30):
        train = trend_error.iloc[:(len(trend_error) - (365 - day) * 24)]
        test = trend_error.iloc[(len(trend_error) - (365 - day) * 24):]
        predictions = pred_svr_model(train, timesteps=(24 * 7) + 1, pred_count=24 * 3)
        plt.plot(predictions.values)
        plt.plot(test.head(24 * 3).values)
        plt.show()




def lstm_forecast(input_frame, n_lookback=24, pred_count=24):
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(input_frame)

    # generate the training sequences
    n_forecast = 1

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=128, validation_split=0.2, verbose=0)

    # generate the multi-step forecasts
    y_future = []

    x_pred = X[-1:, :, :]  # last observed input sequence
    y_pred = Y[-1]  # last observed target value

    for i in range(pred_count):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred.flatten()[0])

    # transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 1)
    y_future = scaler.inverse_transform(y_future)
    return y_future


def test_lstm_preds():
    ercot = build_ercot_load()

    mstl = MSTL(ercot["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    trend_error = pd.DataFrame(res.trend + res.resid, columns=["ERCOT"])

    for day in range(3):
        train = trend_error.iloc[:(len(trend_error) - (365 - day) * 24)]
        test = trend_error.iloc[(len(trend_error) - (365 - day) * 24):]
        predictions = lstm_forecast(train)
        plt.plot(predictions.values)
        plt.plot(test.head(24 * 3).values)
        plt.show()


if __name__ == '__main__':
    ercot = build_ercot_load()
    ercot = ercot[ercot["DATETIME"].dt.year < 2023]

    mstl = MSTL(ercot["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    trend_error = pd.DataFrame(res.trend + res.resid, columns=["ERCOT"])

    predictions = pred_svr_model(trend_error, timesteps=(24 * 7) + 1, pred_count=24 * 3)
    predictions.to_csv("svr_preds.csv")

