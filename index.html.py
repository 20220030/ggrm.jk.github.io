import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Membuat sidebar untuk pengaturan
st.sidebar.title("Pengaturan Data Historis")
start_date = st.sidebar.date_input("Pilih Tanggal Awal", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Pilih Tanggal Akhir", pd.to_datetime("2022-01-01"))
time_period = st.sidebar.selectbox("Time Period", ["1Y", "2Y", "5Y", "Max"])
view_data = st.sidebar.radio("Show Data", ["Historical Prices", "Dividends Only", "Stock Splits", "Capital Gain"])
frequency = st.sidebar.radio("Frequency", ["Daily", "Weekly", "Monthly"])

# Unduh data historis harga saham GGRM.JK dari Yahoo Finance berdasarkan pengaturan
period = time_period
if period == "1Y":
    start_date = "2021-01-01"
elif period == "2Y":
    start_date = "2020-01-01"
elif period == "5Y":
    start_date = "2017-01-01"
elif period == "Max":
    start_date = "2000-01-01"

stock_data = yf.download('GGRM.JK', start=start_date, end='2022-01-01', actions=True, progress=False)

# Ambil kolom harga penutupan (Close) sebagai target prediksi
closing_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalisasi data harga saham
scaler = MinMaxScaler()
scaled_closing_prices = scaler.fit_transform(closing_prices)

# Membagi data menjadi data pelatihan dan pengujian
train_size = int(len(scaled_closing_prices) * 0.8)
train_data = scaled_closing_prices[:train_size]
test_data = scaled_closing_prices[train_size:]

# Menyiapkan data untuk model RNN
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 30  # Jumlah data sebelumnya yang akan digunakan
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Membangun model RNN dengan lapisan LSTM
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(units=1, activation='linear'))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Pelatihan model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluasi model
loss, mae = model.evaluate(X_test, y_test)

# Visualisasi hasil pelatihan
st.title('Stock Price Prediction with RNN')
st.write(f"Loss: {loss:.4f}, Mean Absolute Error: {mae:.4f}")

st.write("Training and Validation Loss")
st.line_chart(pd.DataFrame({'Train Loss': history.history['loss'], 'Validation Loss': history.history['val_loss']}))

st.write("Model Training History")

# Unduh data historis harga saham GGRM.JK berdasarkan pengaturan
if view_data == "Dividends Only":
    data_to_display = stock_data['Dividends']
elif view_data == "Stock Splits":
    data_to_display = stock_data['Stock Splits']
elif view_data == "Capital Gain":
    data_to_display = stock_data['Capital Gain']
else:  # Default: Historical Prices
    data_to_display = stock_data['Close']

if frequency == "Weekly":
    data_to_display = data_to_display.resample("W").last()
elif frequency == "Monthly":
    data_to_display = data_to_display.resample("M").last()

st.write(f"Data Tampilan: {view_data}, Periode: {time_period}, Frekuensi: {frequency}")
st.write(data_to_display)
