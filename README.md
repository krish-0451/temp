# temp
import numpy as np
from scipy.stats import skew, kurtosis

def period_stat(df, period=20, volume_multiplier=2):
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Drop rows with NaN values in the relevant columns
    df.dropna(subset=['High', 'Low', 'Close','Open','Volume'], inplace=True)



    # Compute the required metrics
    df[f'{period}_Max%'] = (df['High'].rolling(window=period).max() * 100) / df['Close']  # Max% relative to Close
    df[f'{period}_Min%'] = (df['Low'].rolling(window=period).min() * 100) / df['Close']   # Min% relative to Close
    df[f'{period}_Range%'] = df[f'{period}_Max%'] - df[f'{period}_Min%']  # Range% relative to Close
    df[f'{period}_Avg_Price%'] = ((df[f'{period}_Max%'] + df[f'{period}_Min%']) / 2)  # Avg Price% relative to Close
    
    # Price Distribution Metrics
    df[f'{period}_Std%'] = df['Close'].rolling(window=period).std() * 100 / df['Close']  # Standard Deviation in % terms
    df[f'{period}_Skew'] = df['Close'].rolling(window=period).apply(lambda x: skew(x))  # Skewness
    df[f'{period}_Kurtosis'] = df['Close'].rolling(window=period).apply(lambda x: kurtosis(x))  # Kurtosis
    df[f'{period}_IQR%'] = (df['Close'].rolling(window=period).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)) / df['Close']) * 100  # IQR in percentage terms
    df[f'{period}_CV'] = df[f'{period}_Std%'] / df['Close'] * 100  # Coefficient of Variation in percentage
    df[f'{period}_Range_to_Std'] = df[f'{period}_Range%'] / df[f'{period}_Std%']  # Range to Standard Deviation ratio
    df[f'{period}_Variance'] = df['Close'].rolling(window=period).var() * 100 / df['Close']  # Variance in percentage terms

    # Trend Change and Reversal Metrics
    # Down days and up days percentage
    df[f'{period}Up_Day'] = (df['Close'] > df['Open']).astype(int)  # 1 for Up day, 0 for Down day
    df[f'{period}Down_Day'] = (df['Close'] < df['Open']).astype(int)  # 1 for Down day, 0 for Up day

    df[f'{period}_Up_Days_Percentage'] = df[f'{period}Up_Day'].rolling(window=period).mean() * 100  # % of Up Days in last 'period'
    df[f'{period}_Down_Days_Percentage'] = df[f'{period}Down_Day'].rolling(window=period).mean() * 100  # % of Down Days in last 'period'

    # Consecutive Up and Down Days
    def consecutive_days(arr, trend_type):
        count, max_count = 0, 0
        for value in arr:
            if value == trend_type:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count
    
    # Apply consecutive day calculation for Up and Down trends
    df[f'{period}_Consecutive_Up_Days'] = df[f'{period}Up_Day'].rolling(window=period).apply(lambda x: consecutive_days(x, 1))
    df[f'{period}_Consecutive_Down_Days'] = df[f'{period}Down_Day'].rolling(window=period).apply(lambda x: consecutive_days(x, 1))

    # Market Movement Cycle Acceleration
    df[f'{period}_Market_Movement_Acceleration%'] = (((df['Close'] - 2 * df['Close'].shift(1) + df['Close'].shift(period)) / (1 ** 2)) / df['Close']) * 100

    # Additional Candle Metrics
    df[f'{period}_Avg_Candle_Size%'] = ((df['High'] - df['Low']).rolling(window=period).mean() / df['Close']) * 100  # Average Candle Size in percentage
    df[f'{period}_Candle_Size_Std%'] = ((df['High'] - df['Low']).rolling(window=period).std() / df['Close']) * 100  # Std of Candle Size in %

    # Candle Body to Wick Ratio
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Candle_Wick_Upper'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Candle_Wick_Lower'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df[f'{period}_Candle_Body_to_Wick_Ratio'] = df['Candle_Body'] / (df['Candle_Wick_Upper'] + df['Candle_Wick_Lower'])

    # Change per Candle (in percentage)
    df[f'{period}_Change_Per_Candle_%'] = (df['Close'] - df['Open']) / df['Open'] * 100

    # Additional Requested Metrics

    # Divergence Percentage: Close vs SMA
    df[f'{period}_Divergence_Percentage'] = ((df['Close'] - df['Close'].rolling(window=period).mean()) / df['Close']) * 100

    # Average Slope
    def average_slope_percentage(x):
        y = x.values
        x = np.arange(len(x))
        slope = np.polyfit(x, y, 1)[0]  # Linear regression to find slope
        return slope / y[-1] * 100  # Normalize slope to percentage of the last closing price

    df[f'{period}_Avg_Slope%'] = df['Close'].rolling(window=period).apply(average_slope_percentage)  # Average Slope in percentage terms


    # Average Bullish Candle Ratio and Average Bearish Candle Ratio
    df[f'{period}_Avg_Bullish_Candle_Ratio'] = df[f'{period}Up_Day'].rolling(window=period).mean() * 100
    df[f'{period}_Avg_Bearish_Candle_Ratio'] = df[f'{period}Down_Day'].rolling(window=period).mean() * 100

    # Average Volume %
    df[f'{period}_Avg_Volume_%'] = (df['Volume'] / df['Volume'].rolling(window=period).mean()) * 100

    # Volume Spike %
    df[f'{period}_Volume_Spike_%'] = (df['Volume'] / df['Volume'].rolling(window=period).mean()) * 100
    df[f'{period}_Volume_Spike_%'] = df[f'{period}_Volume_Spike_%'].where(df[f'{period}_Volume_Spike_%'] > volume_multiplier * 100, 0)

    # Close Lag Percentage
    df[f'{period}_Close_Lag_1%'] = ((df['Close'] - df['Close'].shift(1)) / df['Close']) * 100
    df[f'{period}_Close_Lag_5%'] = ((df['Close'] - df['Close'].shift(5)) / df['Close']) * 100
    df[f'{period}_Close_Lag_20%'] = ((df['Close'] - df['Close'].shift(20)) / df['Close']) * 100

    # Day, Week of the Day, Month, Quarter
    df['Day_of_Week'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
    df['Week_of_Year'] = df.index.isocalendar().week
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

    # Net Change Percentage (Open of First Row and Close of Last Row of the Period)
    df[f'{period}_Net_Change_%'] = ((df['Close'] - df['Open'].shift(period - 1)) / df['Open'].shift(period - 1)) * 100

    # Cumulative Return Percentage
    df[f'{period}_Cumulative_Return_%'] = ((df['Close'] / df['Close'].shift(period - 1)) - 1) * 100

    # Strength Metric: (Close(period) - Close(1)) / (Max - Min)
    df[f'{period}_Strength'] = (df['Close'] - df['Close'].shift(period - 1)) / (df[f'{period}_Max%'] - df[f'{period}_Min%'])


    return df


