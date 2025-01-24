import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# File paths
# File paths
input_file = r"C:\Users\Lenovo\OneDrive\Documents\Projects\Python\Stock prediction\Price pred\CHALET.NS.csv"
output_file = r"C:\Users\Lenovo\OneDrive\Documents\Projects\Python\Stock prediction\Price pred\Modified_CHALET.NS.csv"

def preprocess_csv(file_path, output_file, time_step=100):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

        # Remove commas and convert numeric columns to float
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            df[col] = df[col].replace({',': ''}, regex=True).astype(float)

        # Normalize the 'Close' column
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['Close'] = scaler.fit_transform(df[['Close']])

        # Ensure sufficient data for testing
        if len(df) < time_step + 1:
            raise ValueError("Not enough data for the specified time step.")

        # Save modified CSV
        df.to_csv(output_file, index=False)
        print(f"Modified CSV saved at {output_file}")
        return df, scaler

    except Exception as e:
        print(f"Error during CSV preprocessing: {e}")
        return None, None

def create_dataset(dataset, time_step=100):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Preprocessing
data, scaler = preprocess_csv(input_file, output_file)
if data is None:
    exit()

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data['Close'].values[:train_size].reshape(-1, 1)
test_data = data['Close'].values[train_size:].reshape(-1, 1)

# Prepare training and testing datasets
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Check for sufficient test data
if X_test.shape[0] == 0:
    raise ValueError("Not enough testing data for the specified time step.")

# Reshape for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Print success message
print("Program executed successfully! All errors resolved.")
