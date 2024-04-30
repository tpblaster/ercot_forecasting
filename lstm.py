import json

from statsmodels.tsa.stl.mstl import MSTL
import main
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ercot = main.build_ercot_load()
    train = ercot[ercot["DATETIME"].dt.year < 2023]

    mstl = MSTL(train.set_index("DATETIME")["ERCOT"],
                periods=[24, 24 * 7, 365 * 24], iterate=3, windows=[11, 15, 9999],
                stl_kwargs={"seasonal_deg": 0,
                            "inner_iter": 2,
                            "outer_iter": 0})

    res = mstl.fit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform((res.trend + res.resid).values.reshape(-1, 1))

    # generate the training sequences
    n_forecast = 24 * 3
    n_lookback = 24 * 3

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    X_ = y[- n_lookback:]
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    model.save(f"{n_lookback}-{n_forecast}-lstm-single-forecast.hd5")

    with open(f"{n_lookback}-{n_forecast}-lstm-single-forecast.json", 'w') as f:
        json.dump(model.to_json(), f)
