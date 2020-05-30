#!/usr/bin/env python
# coding: utf-8

# #Friday Special : Predictive Modeling |#DS360withAkanksha
# #Stock Price Prediction

# In[1]:


#Importing libraries

import numpy as np     #To apply mathematical functions & opertations
import matplotlib.pyplot as plt #For Visualization purpose
import pandas as pd #For Data Analysis & Manipulation
import datetime #To work with date & dates observations


# In[2]:


stockdata = pd.read_csv('Google_Stock_Price_Train.csv',index_col="Date",parse_dates=True)


# In[3]:


stockdata.head()


# In[4]:


#To detect missing values

stockdata.isna().any()


# In[5]:


stockdata.info()


# In[6]:


stockdata['Open'].plot(figsize=(15,5))


# In[7]:


#Converting column "a" of a DataFrame

stockdata["Close"] = stockdata["Close"].str.replace(',', '').astype(float)


# In[8]:


stockdata["Volume"] = stockdata["Volume"].str.replace(',', '').astype(float)


# In[9]:


# Calculating 7 days rolling mean in iterations, hence 1st 6 won't be calculated

stockdata.rolling(7).mean().head(20)


# In[12]:


stockdata['Close: 30 Day Mean'] = stockdata['Close'].rolling(window=30).mean()
stockdata[['Close','Close: 30 Day Mean']].plot(figsize=(15,5))


# In[13]:


# specifying a minimum number of periods to be 1 in 30 days period

stockdata['Close'].expanding(min_periods=1).mean().plot(figsize=(15,5))


# In[14]:


training_set=stockdata['Open']
training_set=pd.DataFrame(training_set)


# In[15]:


# Feature Scaling starts here

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# In[16]:


# To forecast the 61st day stockprice, Creating a data structure with 60 timesteps and 1 output

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[19]:


# Phase 2 - LSTM Architecture | Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[20]:


# Initialising the RNN
regressor = Sequential()


# In[21]:


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))


# In[22]:


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[26]:


#Phase 3 - Making Predictions

# Getting the real stock price of 2017
stockdata_test = pd.read_csv('Google_Stock_Price_Test.csv',index_col="Date",parse_dates=True)


# In[27]:


real_stock_price = stockdata_test.iloc[:, 1:2].values


# In[28]:


stockdata_test.head()


# In[29]:


dataset_test.info()


# In[30]:


stockdata_test["Volume"] =stockdata_test["Volume"].str.replace(',', '').astype(float)


# In[31]:


test_set=stockdata_test['Open']
test_set=pd.DataFrame(test_set)


# In[32]:


test_set.info()


# In[33]:


# Getting the predicted stock price of 2017
stockdata_total = pd.concat((stockdata['Open'], stockdata_test['Open']), axis = 0)
inputs = stockdata_total[len(stockdata_total) - len(stockdata_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[34]:


predicted_stock_price=pd.DataFrame(predicted_stock_price)
predicted_stock_price.info()


# In[35]:


# Phase 4 : Visualizing the data

plt.plot(real_stock_price, color = 'green', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[ ]:


# Conclusion : Looks like our model is doing great in making predictions 

