import joblib
import keras.models
import pandas as pd
import tensorflow.python.keras.models
from statsmodels.tsa.stl.mstl import MSTL

from main import build_ercot_load
import statsmodels.api as sm


def create_forecast(trend_resid, window=24 * 3):
    daily_arima = sm.load("day_ercot.pkl")
    weekly_arima = sm.load("week_ercot.pkl")
    lstm_scaler = joblib.load('72-72-lstm-scaler.gz')
    trend_error_lstm = keras.models.load_model("72-72-lstm-single-forecast.hd5")
    yearly_snaive = pd.read_csv("snaive-model.csv", index_col="DATETIME")
    day_forecast = daily_arima.forecast(steps=window).reset_index(drop=True)
    week_forecast = weekly_arima.forecast(steps=window).reset_index(drop=True)
    year_forecast = yearly_snaive.head(window)["yearly_seasonality"].reset_index(drop=True)
    y = lstm_scaler.transform(trend_resid.tail(window).values.reshape(-1, 1))
    error_trend_forecast = trend_error_lstm.predict(y.reshape(1, window, 1)).reshape(-1, 1)
    error_trend_forecast = lstm_scaler.inverse_transform(error_trend_forecast)
    error_trend_forecast = pd.Series(error_trend_forecast.reshape(1, -1)[0])
    total_forecast = pd.concat([day_forecast, week_forecast, year_forecast, error_trend_forecast], ignore_index=True, axis=1)
    total_forecast.to_csv("forecast.csv")


def test_forecast():
    ercot = build_ercot_load()
    train = ercot[ercot["DATETIME"].dt.year < 2023]

    mstl = MSTL(train.set_index("DATETIME")["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()
    create_forecast(res.trend + res.resid)


if __name__ == "__main__":
    trend_error_lstm = keras.models.load_model("72-72-lstm-single-forecast.hd5")
    print()
