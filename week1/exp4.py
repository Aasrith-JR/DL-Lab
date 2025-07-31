import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# data
x = np.linspace(0, 1, 1000)
y = 5 * x + 7 + np.random.randn(1000)

# build the architecture
model = Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

# compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# Build a model
res = model.fit(x, y, epochs=50)

# predict
predict = model.predict(x)

# visualization
plt.scatter(x, y, label='original data', color='blue')
plt.plot(x, predict, label='predictions', color='red')
plt.title("Optimizer: SGD")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()