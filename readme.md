# Backtesting App
App is live [HERE](https://backtesting-app.izem.ovh/)
## Description
This Streamlit app provides a user interface for analyzing strategy returns on financial market data. The users can select a data source (Binance or Yahoo Finance), choose a market, strategy, timeframe, and date period. It then displays the historical returns of the selected strategy on the chosen market.

Note that I prefer to keep my personal strategies private, I simply provide an example in `strategies.py.example` (simply rename it to `strategies.py` if you want to use it in the app).


## Features
* Data Sources: Choose between Binance and Yahoo Finance as the data source (only World Indices are supported for Yahoo Finance so far)
* Market Selection: Select a market from a dynamically loaded list based on the chosen data source
* Strategy Options: Choose from a list of predefined strategies for analyzing market data
* Timeframe Selection: Pick a timeframe for the analysis, including options like 1 hour, 4 hours, 1 day, and 1 week
* Date Range Picker: Select a specific date range for the analysis, with a default period of the last 12 months
* Strategy Returns: View and analyze the returns of the selected strategy, displayed in a table format
* Benchmark vs. Strategy Cumulative Returns: Visualize the cumulative returns of the selected strategy compared to the benchmark over time
* Monthly Returns: Explore the monthly returns of both the benchmark and the selected strategy through a grouped bar chart
