# End to End Machine Learning Project 5: Satrix Collective Investment Scheme - Satrix MSCI World Feeder Portfolio (STXWDM.JO)  With Neural Networks
#--------------------------- Brief on Long Short Term Networks------------------------------#

#--------------------------- Brief on Long Short Term Networks------------------------------#

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf



#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10


#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


import datetime
from datetime import date, timedelta
today = date.today()


d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2


data = yf.download('STXWDM.JO', start=start_date, end=end_date, progress=False)

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
data.tail()

print(data.head())
print('-------------------------------------------------------------')
print(data.tail())
#print(data.columns)

print('-------------------------------------------------------------')
#plot the graph
plt.figure(figsize=(16,8))
plt.plot(data["Adj Close"], label='Adj Close Price history')

######################### Satrix MSCI ######################


plt.figure(figsize=(10,4))
plt.title(' Satrix MSCI')
plt.xlabel('Date')
plt.ylabel('Close')
plt.plot(data["Adj Close"])
plt.show()


#correlation = data.corr()
#plt.figure(figsize=(15,15))
#plt.title('Correlation Matrix')
#sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
#plt.show()


correlation = data.corr()
print(correlation['Adj Close'].sort_values(ascending=False))
sns.heatmap(correlation)
plt.show()


x = data[["Open", "High", "Low", "Volume"]]
y = data["Adj Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

validation_size = 0.2
train_size = int(len(x) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
