from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

#Load Data
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Prepocessing
y_train=to_categorical(y_train)
y_actuals = y_test
y_test=to_categorical(y_test)

#Build the architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#Compile
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#Train
result=model.fit(x_train,y_train,epochs=1,batch_size=32,validation_data=(x_test,y_test))
print(result.history.keys())
print(result.history.items())

#Evaluate
loss,accuracy=model.evaluate(x_test,y_test)
print(f"test loss:{loss},\ntest_accuracy:{accuracy}")

#Predict
predictions = model.predict(x_test)
predictedLabel = np.argmax(predictions, axis=1)
print(f"Predicted label: {(predictedLabel[10])}")

#Print the predicted image
for image in range(5):
    print(f"Act: {y_test[image]} \n Pred: {predictedLabel[image]}")
    plt.subplot(1,5,image+1, title=f"Actual: {y_actuals[image]} \n Predicted: {predictedLabel[image]}")
    plt.imshow(x_test[image])
    plt.xticks([])
    plt.yticks([])
plt.show()

#Visualization
plt.plot(result.history['loss'],label='train loss',color='blue')
plt.plot(result.history['val_loss'],label='valdation loss',color='red')
plt.xticks(np.arange(1, 50, 2))
plt.xlabel("Epochs")
plt.yticks(np.arange(1, 50, 2))
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()


plt.plot(result.history['accuracy'],label='train accuracy',color='blue')
plt.plot(result.history['val_accuracy'],label='valdation accuracy',color='red')
plt.xticks(np.arange(1, 50, 2))
plt.yticks(np.arange(1, 50, 2))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()
