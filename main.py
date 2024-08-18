import numpy as np
import pandas as pd
import matplotlib
import tkinter
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = "AAPL" 

ticker = yf.Ticker(company)
history = ticker.history(period='max')

if history.empty:
    raise ValueError(f"No data found for ticker symbol {company}. Please check the symbol and try again.")

start = history.index[0] 
end = dt.datetime.now()

data = yf.download(company , start , end)

# Prepare data
scaler  = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x ,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units= 1))   # Prediction of next closest value

model.compile(optimizer = "adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)


''' Test the model accuracy on the existing data'''

test_start = dt.datetime(2024,1,1)
test_end = dt.datetime.now()

test_data = yf.download(company,test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis= 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
# x_test = x_test.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the Test Predictions

plt.plot(actual_prices, color="black", label= f"Actual {company} Prices")
plt.plot(predicted_prices, color="red", label=f"Predicted {company} Prices")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()


# Predict the next data
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs + 1) , 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))


prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

