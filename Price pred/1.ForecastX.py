import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = r'C:\Users\Lenovo\OneDrive\Documents\Projects\Python\Stock prediction\Price pred\CHALET_30min_daily_filtered.csv'
df = pd.read_csv(file_path)

# Ensure 'Date' column is parsed as datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate 50 and 200-period moving averages using pandas
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the candlesticks
for i in range(len(df)):
    if df['Close'][i] >= df['Open'][i]:
        color = 'green'  # Bullish (Close >= Open)
        ax.plot([df.index[i], df.index[i]], [df['Low'][i], df['High'][i]], color=color, lw=1)  # Line for high-low
        ax.plot([df.index[i], df.index[i]], [df['Open'][i], df['Close'][i]], color=color, lw=6)  # Green rectangle
    else:
        color = 'red'  # Bearish (Close < Open)
        ax.plot([df.index[i], df.index[i]], [df['Low'][i], df['High'][i]], color=color, lw=1)  # Line for high-low
        ax.plot([df.index[i], df.index[i]], [df['Open'][i], df['Close'][i]], color=color, lw=6)  # Red rectangle

# Plot the moving averages
ax.plot(df.index, df['MA50'], label='50-period MA', color='blue', lw=1)
ax.plot(df.index, df['MA200'], label='200-period MA', color='red', lw=1)

# Add title and labels
ax.set_title('Candlestick Chart with 50 and 200 Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Add a legend for moving averages
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
