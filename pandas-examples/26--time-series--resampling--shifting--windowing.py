import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas as pd
import yfinance as yfin


yfin.pdr_override()

# Load part of the S&P 500 price history
sp500 = pdr.data.get_data_yahoo('^GSPC', start='2018-01-01', end='2023-01-01')
sp500.head()
#                    Open         High          Low        Close    Adj Close      Volume
# Date
# 2018-01-02  2683.729980  2695.889893  2682.360107  2695.810059  2695.810059  3397430000
# 2018-01-03  2697.850098  2714.370117  2697.770020  2713.060059  2713.060059  3544030000
# 2018-01-04  2719.310059  2729.290039  2719.070068  2723.989990  2723.989990  3697340000
# 2018-01-05  2731.330078  2743.449951  2727.919922  2743.149902  2743.149902  3239280000
# 2018-01-08  2742.669922  2748.510010  2737.600098  2747.709961  2747.709961  3246160000

sp500 = sp500['Close']
sp500.plot()
# plt.show()
plt.savefig('../pandas-examples-figures/sp500-1.svg')
plt.close()

# resampling

# Resample by using:
# - the 'resample' method is fundamentally a data aggregation;
# - the 'asfreq' method is fundamentally a data selection.

sp500.plot(alpha=0.5, style='-')
sp500.resample('BA').mean().plot(style=':')  # average of the previous year
sp500.asfreq('BA').plot(style='--')  # value at the end of the year
plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
# plt.show()
plt.savefig('../pandas-examples-figures/sp500-2.svg')
plt.close()

# Resample the business day data at a daily frequency (i.e., including weekends)
fig, ax = plt.subplots(2, sharex=True)
data = sp500.iloc[:30]  # first 30 rows
data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(["back-fill", "forward-fill"])
# plt.show()
plt.savefig('../pandas-examples-figures/sp500-3.svg')
plt.close()

# time shifts

# Resample the data to daily values, and shift by 365 to compute
# the 1-year return on investment for the S&P 500 over time

sp500 = sp500.asfreq('D', method='pad')
ROI = 100 * (sp500.shift(-365) - sp500) / sp500
ROI.plot()
plt.ylabel('% Return on Investment after 1 year')
# plt.show()
plt.savefig('../pandas-examples-figures/sp500-4.svg')
plt.close()

# rolling windows (rolling statistics, e.g. rolling mean, or running average)

# one-year centered rolling mean and standard deviation of the stock prices
rolling = sp500.rolling(365, center=True)
data = pd.DataFrame(
    {
        'input': sp500,
        'one-year rolling mean': rolling.mean(),
        'one-year rolling median': rolling.median()
    }
)
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)
# plt.show()
plt.savefig('../pandas-examples-figures/sp500-5.svg')
plt.close()
