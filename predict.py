import numpy as np
import tensorflow as tf
from tensorflow import keras
# import the necessary packages
from sklearn.model_selection import train_test_split
from utils import datasets
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt
import cv2

# It can be used to reconstruct the model identically.


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to model")
ap.add_argument("-s", "--size", type=int, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

size = args["size"]

# restruct model
model = keras.models.load_model(args["model"])

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])

# Example of a picture

cv2.imshow("test",images[0])
cv2.waitKey(0)

num_images = 10
for index in range(0, num_images):
	plt.subplot(2, (num_images//2), index + 1)
	plt.imshow(images[index], interpolation='nearest')
	plt.title(f'{df["price"][index]}')
	plt.axis('off')
plt.show()


# image normalization
images = images / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "tr_TR.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))