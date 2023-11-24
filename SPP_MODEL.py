# IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
import datetime

# ACQUIRING THE DATA SET FROM STOOQ
start =  datetime.datetime(1999, 1, 1)
end =  datetime.datetime(2022, 12, 31)
df = data.DataReader('AAPL','stooq',start,end)
df.head()

# LET'S VISULAIZE THE DATA
plt.plot(df.Close)

# MOVING AVERAGE IS ESSENTIAL FOR THE PREDICTION OF THE VALUE HERE WE TAKE FOR 100 DAYS
ma100 = df.Close.rolling(100).mean()
ma100

# LET'S VISULIZE ACTUAL VALUES AGAINST MOVING AVERAGE PREDICTED VALUES
plt.figure(figsize=(16,14))
plt.plot(df.Close)
plt.plot(ma100,'r')

# MOVING AVERAGE FOR  200 DAYS
ma200 = df.Close.rolling(200).mean()
ma200

# PLOT AND COMPARE CLOSING, 100MA AND 200 MA
plt.figure(figsize=(16,14))
plt.plot(df.Close,'b')
plt.plot(ma100,'y')
plt.plot(ma200,'r')
plt.xlabel('DAYS')
plt.ylabel('VALUES')
plt.title('CLOSING VALUE vs 100 DAYS MA vs 200 DAYS MA')

# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.700)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.700):int(len(df))])

# Let's see how our data is divided...
print("",data_training.shape,"\n",data_testing.shape)

# We've to scale down the data for the values b/w 0 &1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array

# Most importatant step...

x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

# LET US BUILD AN LSTM(LONG-SHORT TERM MEMORY) DEEPLEARNING MODEL

# IMPORTING LIBRARIES
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential


model = Sequential()
#LSTM LAYER
model.add(LSTM(units=50,activation='tanh',return_sequences = True,
                   input_shape=(x_train.shape[1],1)))
#TO AVOID OVERFITTING
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='tanh',return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='tanh',return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='tanh'))
model.add(Dropout(0.5))

# FULLU CONNECTED LAYER, CONNECTS ALL THE NEURAL NETWORK LAYERS.
model.add(Dense(units = 1))

# MODEL SUMMARY
model.summary()

#OPTIMIZING AND FITTING THE MODEL WITH TRAINING DATA
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

# SAVE MODEL FOR FUTURE PURRPOSES
model.save('keras_model.h5')

# LET'S TEST THE MODEL
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df.head()

input_data = scaler.fit_transform(final_df)
input_data

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

# Model Prediction

y_predicted = model.predict(x_test)

scale_factor  = 1/ 0.41569342
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
