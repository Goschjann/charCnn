import keras
import numpy as np
import pandas
import datetime
import math
import gc
import random

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import MaxPooling1D,  Dropout, Flatten, Dense
from keras.layers import Conv1D
from keras.optimizers import SGD

import projectlib as pl
gc.enable()
gc.get_objects()

# parameters

maxChars = 1014



pl.check()
# path to train data
#dataPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTrain.csv"

# Small Yelp polarity data set by Le Cunn et al
dataPath = '/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_polarity/trainPrep.csv'
alphabetPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt"

lenAlpha = len(open(alphabetPath).read())


# divide IDs in valid and train IDs
lenAllData = 560000
allIDs = range(0, lenAllData)


trainIDs = random.sample(allIDs, math.ceil(0.8 * lenAllData))
validIDs = np.setdiff1d(allIDs, trainIDs)

# dictionary that stores IDs for train and validation
# necessary for data generator
partition = {"train": trainIDs, "validation": validIDs}



# RUNNING VERSION



#######################
#
#       Build Model
#
#######################


model = keras.models.Sequential()

# 1st layer
# try different input shapes
model.add(Conv1D(filters=256, kernel_size=7, activation="relu", padding="valid", input_shape=(maxChars, lenAlpha)))
model.add(MaxPooling1D(3))

model.add(Conv1D(256, 7, activation="relu"))

model.add(MaxPooling1D(3))

model.add(Conv1D(256, 3, activation="relu"))

model.add(Conv1D(256, 3, activation="relu"))

model.add(Conv1D(256, 3, activation="relu"))

model.add(Conv1D(256, 3, activation="relu"))

model.add(MaxPooling1D(3))

# this is what "Flatten" tells us
amount_total_features = math.ceil(maxChars/ 9) * 256

model.add(Flatten())

model.add(Dense(1024, activation="relu"))

model.add(Dropout(rate=0.5))

model.add(Dense(1024, activation="relu"))

model.add(Dropout(rate = 0.5))

model.add(Dense(2, activation="softmax"))

print(model.summary())

############ train trial

# TODO: try binary_crossentropy and change buildSetDG in pl accordingly

model.compile(loss = "categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("Train that homeboy")

# TODO not elegant
batch_size = 1000


params = {"dim_x": lenAlpha,
          "dim_y": maxChars,
          "batch_size": batch_size,
          "shuffle": True,
          "path_data": dataPath,
          "path_alphabet": alphabetPath,
          "maxChars": maxChars,
          "allIDs": allIDs}

# initalize the train and validation data generator
trainGenerator = pl.DataGenerator(**params).generate(partition['train'])
validGenerator = pl.DataGenerator(**params).generate(partition['validation'])



# callbacks during training
stop_train = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                           min_delta = 0,
                                           patience = 4,
                                           verbose = 0,
                                           mode = 'auto')

csvLogger = keras.callbacks.CSVLogger("charCnn_7_polarity_log.csv")


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

model.save("charCnn_7_polarity.h5")

print("\n and done with saving")

