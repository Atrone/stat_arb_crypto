# Define the backtest results and BTC costs
backtest_data = [
    {"profit": 2381.7747742142883, "hours": 250, "btc_cost": 40000},
    {"profit": -886.2146842064366, "hours": 100, "btc_cost": 40000},
    {"profit": -1000.00, "hours": 100, "btc_cost": 40000},
    {"profit": 2044.2080660286929, "hours": 100, "btc_cost": 58000},
    {"profit": 6000.00, "hours": 100, "btc_cost": 58000},
    {"profit": -1350.00, "hours": 100, "btc_cost": 69000},
    {"profit": 967.7458176597763, "hours": 100, "btc_cost": 69000},
    {"profit": -207.2085659829528, "hours": 100, "btc_cost": 69000},
    {"profit": -1462.2612239729012, "hours": 100, "btc_cost": 69000},
    {"profit": 717.00, "hours": 50, "btc_cost": 42000},
    {"profit": 115.59741495211688, "hours": 100, "btc_cost": 42000},
    {"profit": 12393.42631087813, "hours": 250, "btc_cost": 100000}
]

# Define the daily handicap
handicap_per_day = 0.003

# Calculate the daily profit percentages
results = []
for entry in backtest_data:
    profit = entry["profit"]
    hours = entry["hours"]
    btc_cost = entry["btc_cost"]

    # Calculate profit as a percentage of BTC cost
    profit_percentage = (profit / btc_cost) * 100

    # Convert hours to days
    days = hours / 24

    # Calculate daily profit percentage
    daily_profit_percentage = profit_percentage / days

    # Remove handicap
    adjusted_daily_profit_percentage = daily_profit_percentage + handicap_per_day * days

    # Store the result
    results.append(adjusted_daily_profit_percentage)

print(results)
# Calculate the average daily profit percentage
average_daily_profit_percentage = sum(results) / len(results)

# Add numpy import for statistical calculations
import numpy as np

# Calculate standard deviation of daily profit percentages
std_dev = np.std(results)

# Calculate Z-score for 6%
target_percentage = 4.0
z_score = (target_percentage - average_daily_profit_percentage) / std_dev

print(f"Average daily profit percentage (without handicap): {average_daily_profit_percentage:.2f}%")
print(f"Standard deviation: {std_dev:.2f}%")
print(f"Z-score for {target_percentage}%: {z_score:.2f}")
