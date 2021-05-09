# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from utils import datasets
from utils import models
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt
import codecs

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
ap.add_argument("-s", "--size", type=int, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())
size = args["size"]

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"], size=size)
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

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
model = models.create_cnn(size*2, size*2, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 10000)
#opt = SGD(lr=1e-3)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
history = model.fit(x=trainImagesX, y=trainY, 
    validation_data=(testImagesX, testY),
    epochs=2000, batch_size=16)

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

model.save(f"model/mymodel-{size}")

plt.figure(1)
plt.title("Loss")
plt.plot(range(len(history.history["loss"])), history.history["loss"])
plt.savefig(f"model/mymodel-{size}/loss.png")

plt.figure(2)
plt.plot(range(len(history.history["val_loss"])), history.history["val_loss"])
plt.title("Val Loss")
plt.savefig(f"model/mymodel-{size}/val_loss.png")

with open(f"model/mymodel-{size}/maxPrice.txt", "w") as f:
	f.write(str(maxPrice))

with codecs.open(f"model/mymodel-{size}/model.txt", "w", "utf-8") as f:
	f.write("[INFO] avg. house price: {}, std house price: {}\n".format(
		locale.currency(df["price"].mean(), grouping=True),
		locale.currency(df["price"].std(), grouping=True)))
	f.write("[INFO] mean: {:.2f}%, std: {:.2f}%\n".format(mean, std))