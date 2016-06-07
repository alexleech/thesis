import pytz
from datetime import datetime

import zipline
from zipline.api import order, record, symbol
from zipline.api import get_datetime
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo

# Load data manually from Yahoo! finance
start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
data = load_bars_from_yahoo(stocks=['AAPL'], start=start,
                            end=end)

# Define algorithm
def initialize(context):
    pass

def handle_data(context, data):

	order(symbol('AAPL'), 10)
	record(AAPL=data[symbol('AAPL')].price)
	#print(get_datetime())
# Create algorithm object passing in initialize and
# handle_data functions
algo_obj = TradingAlgorithm(initialize=initialize,
                            handle_data=handle_data)

# Run algorithm
perf_manual = algo_obj.run(data)
