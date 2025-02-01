import statistics
import matplotlib.pyplot as plt
import pandas as pd
file_path = "C:\\Users\\HP\\Downloads\\Lab Session Data.xlsx"
stock_data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

prices = stock_data.iloc[:, 3]
mean_price = statistics.mean(prices)
var_price = statistics.variance(prices)

wed_prices = stock_data[stock_data.iloc[:, 1] == "Wednesday"].iloc[:, 3]
mean_wed = statistics.mean(wed_prices) if not wed_prices.empty else None

apr_prices = stock_data[pd.to_datetime(stock_data.iloc[:, 0]).dt.month == 4].iloc[:, 3]
mean_apr = statistics.mean(apr_prices) if not apr_prices.empty else None

chg = stock_data.iloc[:, 8]
loss_prob = sum(chg < 0) / len(chg)
profit_wed = sum((chg > 0) & (stock_data.iloc[:, 1] == "Wednesday")) / sum(stock_data.iloc[:, 1] == "Wednesday") if sum(stock_data.iloc[:, 1] == "Wednesday") > 0 else None
cond_prob = profit_wed / (sum(stock_data.iloc[:, 1] == "Wednesday") / len(stock_data)) if profit_wed else None

plt.scatter(stock_data.iloc[:, 1], chg)
plt.xlabel("Day")
plt.ylabel("Chg%")
plt.title("Chg% vs Day")
plt.show()
