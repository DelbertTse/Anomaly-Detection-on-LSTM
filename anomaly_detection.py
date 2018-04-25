import numpy as np
np.random.seed(1018)
import matplotlib.pyplot as plt
import random as rm
from keras.layers import Dense,LSTM,GRU,TimeDistributed
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam

BATCH_START = 0
BATCH_SIZE = 50
TIME_STEPS = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.001

x = np.arange(0,200,0.1)
indice = x.shape[0]
y = rm.uniform(-0.5,0.5)+np.sin(x)
y = np.round(y,3)

plt.figure(figsize=(100,30))
plt.xlim(x.min()*1.1,x.max()*1.1,)
plt.ylim(y.min()*1.1,y.max()*1.1)
plt.plot(x,y,"bo-",linewidth=2, markersize=5)


def getbatch(x,y):
	global BATCH_START,TIME_STEPS
	xs = x[BATCH_START:BATCH_START+BATCH_SIZE*TIME_STEPS].reshape(BATCH_SIZE,TIME_STEPS)
	seq = y[BATCH_START:BATCH_START+BATCH_SIZE*TIME_STEPS].reshape(BATCH_SIZE,TIME_STEPS)
	res = y[BATCH_START+1:BATCH_START+BATCH_SIZE*TIME_STEPS+1].reshape(BATCH_SIZE,TIME_STEPS)
	BATCH_START += TIME_STEPS
	if BATCH_START+BATCH_SIZE*TIME_STEPS >= xs.shape[0]:
		BATCH_START = 0
	return seq[:,:,np.newaxis],res[:,:,np.newaxis],xs

model = Sequential()
model.add(LSTM(CELL_SIZE,batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),return_sequences=True,stateful=True))
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
optimizer = Adam(lr=LR)

model.compile(optimizer=optimizer,loss='mse')

print('-'*50,'training','-'*50)
for step in range(501):
	X_batch,Y_batch,xs = getbatch(x,y)
	cost = model.train_on_batch(X_batch,Y_batch)
	if step % 10 == 0:
		print("train cost:",cost)

def prediction(x_batch):
	pred = model.predict(x_batch,batch_size=x_batch.shape[0],verbose=False)
	return pred

BATCH_START = 0
sequence = []
x_batch,y_batch,xs = getbatch(x,y)
sequence.append(x_batch.flatten()[:TIME_STEPS])
for i in range(49):
	x_batch = prediction(x_batch)
	sequence.append(x_batch.flatten()[:TIME_STEPS])
sequence_y_data = np.array(sequence).flatten()
# sequence_x_data = x[0:sequence_y_data.shape[0]]

# # plt.plot(x[0:sequence_y_data.shape[0]],sequence_y_data,'b--')
# x = x[0:BATCH_SIZE]
# y = sequence_y_data[0:BATCH_SIZE]
# print(x.shape,y.shape)
# # plt.plot(x[0:BATCH_SIZE],sequence_y_data[0:BATCH_SIZE],'b--')
# plt.scatter(x,y)
# plt.show()