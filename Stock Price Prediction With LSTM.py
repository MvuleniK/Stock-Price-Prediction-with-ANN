# End to End Machine Learning Project 4
#--------------------------- Brief on Long Short Term Networks------------------------------#

#--------------------------- Brief on Long Short Term Networks------------------------------#




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go


import datetime
from datetime import date, timedelta
today = date.today()


d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2


data = yf.download('AAPL', start=start_date, end=end_date, progress=False)

data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
data.tail()

print(data.tail())
print(data.columns)


############### Needs rework #####################################
#import plotly.graph_objects as go
#figure = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["AAPL.Open"], high=data["AAPL.High"], low=data["AAPL.Low"], close=data["AAPL.Adj Close"])])
#figure.update_layout(title="Apple Stock Price Analysis", xaxis_rangeslider_visible=False)
#figure.show()

#print(figure.show())

#################################################################


################## Here we show the correlation between the variables #####################
#correlation = data.corr()
#print(correlation['Adj Close'].sort_values(ascending=False))
#sns.heatmap(correlation)
#plt.show()

################# Now we prepare the data for training using Numpy for the manipulation###########
# Numpy is used for the array manipulation with out changing the data
# Sklearn though used for regression & classification algorithms, we are utilising the training
#feature. Thus, it can be used for the neural network training data


x = data[["Open", "High", "Low", "Volume"]]
y = data["Adj Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)



################# Now we prepare the neural network architecture#######################

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)
model.summary()
