
import keras
import numpy as np
import pandas
import datetime
import math
import gc
import random

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.optimizers import SGD

import projectlib as pl
gc.enable()
gc.get_objects()

# parameters

maxChars = 1014




pl.check()
# path to train data
dataPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTrain.csv"
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"

lenAlpha = len(open(alphabetPath).read())


# divide IDs in valid and train IDs
lenAlldata = 194544
allIDs = range(0, lenAlldata)


trainIDs = random.sample(allIDs, math.ceil(0.8 * lenAlldata))
validIDs = np.setdiff1d(allIDs, trainIDs)

# dictionary that stores IDs for train and validation
# that is the interesting one for us
partition = {"train": trainIDs, "validation": validIDs}
# dictionary that stores matching labels, not needed in this case
# labels = {allIDs: dataSent.iloc[:, 1]}




#######################
#
#       Build Model
#
#######################




####################################
#               purzelrakete
# https://github.com/purzelrakete/char-cnn
#
#####################################



######################################
#           Julian Fisher
######################################



filterdepth = 256


"""
filters: specify shape of output from the convolution. We want 256 feature maps with dimensions 70 * 1014
    CHECK: often only depth of the thiny mentioned (256 und fertig)
kernel_size: shape of the (256) filter itself. Our fake 2D conv with same padding sees all character 
    representations ( = lenAlpha) and 

strides = (lenAlpha, 7) due to Julian Fuchs: https://github.com/JulianFuchs/TwitterSentimentCharCNN/blob/master/CharCNNModel.py    
"""
convpad = "same"

model = keras.models.Sequential()

# 1st layer
model.add(Conv2D(filters=filterdepth,
                 # filters = (lenAlpha, maxChars, 1, 256),
                 kernel_size=(lenAlpha, 7),
                 strides=(lenAlpha, 1),
                 padding="same",
                 activation="relu",
                 input_shape=(lenAlpha, maxChars, 1)))

# print(model.summary())

# pool_size=(1,3) pooled nur in y richtung
# makes padding difference? Nope, not in dimensions
# TODO: evaluate that
model.add(MaxPooling2D(pool_size=(1, 3),
                       padding="valid"))


# 2nd layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(1, 7),
                 strides=(1, 1),
                 padding="same",
                 activation="relu"))

# print(model.summary())

# 3rd layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 padding="same",
                 activation="relu"))

# 4th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 padding="same",
                 activation="relu"))

# 5th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 padding="same",
                 activation="relu"))

# 6th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 padding="same",
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(1, 3),
                       padding="same"))

# this is what "Flatten" tells us
amount_total_features = math.ceil(maxChars/ 9) * filterdepth

model.add(Flatten())

model.add(Dense(1024, activation="relu"))

model.add(Dropout(rate=0.5))

model.add(Dense(1024, activation="relu"))

model.add(Dropout(rate = 0.5))

model.add(Dense(1, activation="sigmoid"))

print(model.summary())

############ train trial

model.compile(loss = "binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("Train that homeboy")
# csvLogger = keras.callbacks.CSVLogger("charCnn_1.log")

#model.fit(x = xtrain, y = y_train,
#          callbacks=[csvLogger],
#          batch_size= 100,
#          epochs= 2,
#          verbose=2,
#          shuffle=True,
#          validation_data=(xvalid, y_valid))

# params for data generator
# dim_x = lenAlpha, dim_y = maxChars, batch_size = batch_size

# TODO not elegant
batch_size = 200


params = {"dim_x": lenAlpha,
          "dim_y": maxChars,
          "batch_size": batch_size,
          "shuffle": True,
          "path_data": dataPath,
          "path_alphabet": alphabetPath,
          "maxChars": maxChars,
          "allIDs": allIDs}


trainGenerator = pl.DataGenerator(**params).generate(partition['train'])
validGenerator = pl.DataGenerator(**params).generate(partition['validation'])



# callbacks during training
stop_train = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           min_delta = 0,
                                           patience = 7,
                                           verbose = 0,
                                           mode = 'auto')

csvLogger = keras.callbacks.CSVLogger("charCnn_4.log")


# fit the thingey
model.fit_generator(generator=trainGenerator,
                    steps_per_epoch=len(partition["train"])// batch_size,
                    #steps_per_epoch=10,
                    epochs= 60,
                    verbose= 2,
                    callbacks=[csvLogger, stop_train],
                    validation_data=validGenerator,
                    validation_steps=len(partition["validation"])// batch_size)
                    #validation_steps=5)


print("\n done with trainig ")

model.save("charCnn_4.h5")

print("\n and done with saving")


"""


##################################
#       according to paper
##################################

## parameters that occur often
filterdepth = 256

model = keras.models.Sequential()

"""
#filters: specify shape of output from the convolution. We want 256 feature maps with dimensions 70 * 1014
#    CHECK: often only depth of the thiny mentioned (256 und fertig)
#kernel_size: shape of the (256) filter itself. Our fake 2D conv with same padding sees all character
#    representations ( = lenAlpha) and

# strides = (lenAlpha, 7) due to Julian Fuchs: https://github.com/JulianFuchs/TwitterSentimentCharCNN/blob/master/CharCNNModel.py
"""
convpad = "same"

# 1st layer
model.add(Conv2D(filters=filterdepth,
                 # filters = (lenAlpha, maxChars, 1, 256),
                 kernel_size=(lenAlpha, 7),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu",
                 input_shape=(lenAlpha, maxChars, 1)))

# print(model.summary())

# pool_size=(1,3) pooled nur in y richtung
# makes padding difference? Nope, not in dimensions
# TODO: evaluate that
model.add(MaxPooling2D(pool_size=(1, 3),
                       padding="valid"))

# 2nd layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(lenAlpha, 7),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(1, 3),
                       padding="valid"))

print(model.summary())

# 3rd layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(lenAlpha, 3),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu"))

# 4th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(lenAlpha, 3),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu"))

# 5th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(lenAlpha, 3),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu"))

# 6th layer
model.add(Conv2D(filters=filterdepth,
                 kernel_size=(lenAlpha, 3),
                 strides=(1, 1),
                 padding=convpad,
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(1, 3),
                       padding="valid"))

print(model.summary())

# add fully connected layers, therfore flatten output from previous Convlayer

model.add(Flatten())

# 7th layer
#
model.add(Dense(1024, activation="relu"))
model.add(Activation("relu"))
# 8th layer
# for 2 classes
model.add(Dense(1024))
model.add(Activation("relu"))

# 9th layer
# for 2 classes
model.add(Dense(2, activation="relu"))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accurary"])

print("Train that homeboy")
csvLogger = keras.callbacks.CSVLogger("charCnn_1.log")

model.fit(x=x_train, y=y_train,
          callbacks=[csvLogger],
          batch_size=100,
          epochs=5,
          verbose=2,
          shuffle=True,
          validation_data=(x_valid, y_valid))




"""




