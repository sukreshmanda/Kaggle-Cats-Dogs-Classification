import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from PIL import Image

def load_previous_dataset():
	x = np.load("data.npy")
	y = np.load("labels.npy")
	print("loading from previous data")
	return x,y
	
def create_training_dataset(IMG_X,IMG_Y,SIZE):
	training_data = []
	for category in Categories:
		path = os.path.join(DataDir,category)
		class_num = Categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array,(IMG_X,IMG_Y))
				training_data.append([new_array,class_num])
			except Exception:
				pass
	random.shuffle(training_data)
	x = []
	y = []
	for feature,label in training_data:
		x.append(feature)
		y.append(label)
	x = np.array(x)
	x.reshape(len(x),60,60)
	y = np.array(y)
	np.save("data.npy",x)
	np.save("labels.npy",y)
	return x,y		
	
def create_network(training_data, training_labels):

	import tensorflow as tf
	from keras.models import Sequential
	from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

	training_data = training_data/255.0

	model = Sequential()

	model.add(Conv2D(32,(3,3),input_shape = (60,60,1)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(32,(3,3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Flatten())
	model.add(Dense(32))

	model.add(Dense(1))
	model.add(Activation("sigmoid"))

	model.compile(loss="binary_crossentropy",optimizer = "adam",metrics=["accuracy"])
	model.fit(training_data,training_labels,batch_size=3,validation_split=0.1,epochs=10)
	model.save("saved_model")
	return model
	
def load_previous_train():
	from keras.models import load_model
	model = load_model("saved_model")
	return model
		
 

DataDir = "PetImages"
Categories = ["Cat","Dog"]

training_data,training_labels = create_training_dataset(60,60,10)
training_data = training_data.reshape(len(training_data),60,60,1)

print((training_data.shape),len(training_labels))
#model = create_network(training_data,training_labels)
#model.save("saved_model")
from keras.models import load_model
model = load_model("saved_model")


for i in range(10):
	j = random.randrange(1,20000)
	img = training_data[j].reshape(60,60)
	label = training_labels[j]
	label_pr = model.predict(training_data[j:j+1])
	plt.imshow(img, cmap="gray")
	plt.title(Categories[np.int16(label_pr.reshape(1)).item()]+" "+Categories[np.int16(label.reshape(1)).item()])
	plt.show()
