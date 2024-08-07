import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Load dataset
def load_data(filepath):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(filepath)

# Prepare product data
def prepare_product_data(data):
    """
    Prepare product data by creating a unique product ID for each product and merging with the original data.
    """
    Product = data[['item_name', 'item_brand', 'item_main_category', 'item_sub_category']].drop_duplicates()
    Product.insert(0, 'Product_id', range(1, 1 + len(Product)))
    return pd.merge(data, Product, on=['item_name', 'item_brand', 'item_main_category', 'item_sub_category'], how='inner')

# Aggregate data
def aggregate_data(data):
    """
    Aggregate sales data by transaction date and product ID.
    """
    return data.groupby(['transaction_date', 'Product_id']).agg({
        'item_quantity': 'sum',
        'item_coupon': 'sum'
    }).reset_index()

# Plot data
def plot_data(data, title, xlabel, ylabel):
    """
    Plot item quantities over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data['transaction_date'], data['item_quantity'], linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Normalize data
def normalize_data(data):
    """
    Normalize the dataset using StandardScaler.
    """
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler

# Prepare LSTM data
def prepare_lstm_data(data, n_past, n_future):
    """
    Prepare data for LSTM input by creating sequences of past and future data.
    """
    trainX, trainY = [], []
    for i in range(n_past, len(data) - n_future + 1):
        trainX.append(data[i - n_past:i, :])
        trainY.append(data[i + n_future - 1:i + n_future, 0])
    
    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY

# Build LSTM model
def build_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM model
def train_model(model, trainX, trainY):
    """
    Train the LSTM model with the given data.
    """
    history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)
    
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model

# Make predictions
def make_predictions(model, data, scaler, n_days_for_prediction, train_dates):
    """
    Make future sales predictions using the trained model.
    """
    # Make predictions
    prediction = model.predict(data[-n_days_for_prediction:])
    
    # Repeat the prediction to match the original feature dimensions
    prediction_copies = np.repeat(prediction, data.shape[2], axis=-1)
    
    # Perform inverse transformation
    y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
    
    # Generate future dates for plotting
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction, freq=us_bd).tolist()
    
    return predict_period_dates, y_pred_future

# Plot forecast
def plot_forecast(original, predict_period_dates, y_pred_future):
    """
    Plot original data and forecasted data.
    """
    df_forecast = pd.DataFrame({'transaction_date': np.array(predict_period_dates), 'item_quantity': y_pred_future})
    df_forecast['transaction_date'] = pd.to_datetime(df_forecast['transaction_date'])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=original, x='transaction_date', y='item_quantity', label='Original')
    sns.lineplot(data=df_forecast, x='transaction_date', y='item_quantity', label='Forecast')
    plt.title('Sales Forecast')
    plt.xlabel('Transaction Date')
    plt.ylabel('Item Quantity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return df_forecast

def main():
    # Load and prepare data
    Table2 = load_data('Dataset.csv')
    Table = prepare_product_data(Table2)
    Table_L = aggregate_data(Table)
    
    # Filter specific product data
    df = Table_L[Table_L['Product_id'] == 2081]
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Plot initial data
    plot_data(df, 'Transaction Date vs Item Quantity', 'Transaction Date', 'Item Quantity')
    
    # Normalize and prepare data for LSTM
    df_for_training = df[['item_quantity', 'item_coupon']]
    df_for_training_scaled, scaler = normalize_data(df_for_training)
    trainX, trainY = prepare_lstm_data(df_for_training_scaled, n_past=14, n_future=1)
    
    # Build, train, and predict with the model
    model = build_model((trainX.shape[1], trainX.shape[2]))
    model = train_model(model, trainX, trainY)
    
    # Make predictions
    predict_period_dates, y_pred_future = make_predictions(model, trainX, scaler, n_days_for_prediction=30, train_dates=df['transaction_date'])
    
    # Plot forecast and calculate average quantity
    df_forecast = plot_forecast(df[['transaction_date', 'item_quantity']], predict_period_dates, y_pred_future)
    avg_quantity = df_forecast['item_quantity'].mean()
    print('Average Forecasted Quantity:', avg_quantity)

if __name__ == "__main__":
    main()
