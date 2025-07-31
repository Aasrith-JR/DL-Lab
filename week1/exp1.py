import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# data
x = np.linspace(0, 1, 1000)
y = 5 * x + 7 + np.random.randn(1000)

# build the architecture
model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='linear'))

# compile
model.compile(optimizer='sgd', loss='mse')

# Build a model
model.fit(x, y, epochs=1, verbose=0)

# visualization
plt.scatter(x, y, label='original data')
plt.title("Optimizer: SGD, Epoch: 1")
plt.show()