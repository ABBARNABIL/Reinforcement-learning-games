import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow import keras

'''
    We load the dataset,normalize it and split it into train and test sets.
    We then create a neural network model and train it.
    
'''



X = open("X.txt", "r")
Y = open("Y.txt", "r")

#Getting the X data
#Input size : 63 x 80 x 1
x=[]
lines=0
for line in X : 
    x.extend(map(float,line[:-1].split(",")))
    lines+=1
x = np.reshape(x,(lines//63,63*80))
x /= 255

print("number of images : ", lines//63)

#Getting the Y data (labels)
y = []
for line in Y :
    y.append(int(line[:-1]))
y = np.array(y)

print("number of labels : ", len(y))
nb_classes = 4
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train,y_test = keras.utils.to_categorical(y_train,nb_classes), keras.utils.to_categorical(y_test,nb_classes)

#Creating the model
model = Sequential()
model.add(Dense(units=200,input_dim=63*80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=nb_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])





# fit the model
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")
