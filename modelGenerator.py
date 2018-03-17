from keras.layers import MaxPooling1D,  Dropout, Flatten, Dense
from keras.layers import Conv1D
from keras.utils.vis_utils import plot_model
import projectlib as pl
import keras
import numpy as np
import math
import gc
import random

gc.enable()
gc.get_objects()
pl.check()


# parameters

maxChars = 1014

# path to train data
#dataPath = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTrain.csv"

# Yelp polarity data set by Le Cunn et al
dataPath = '..data/trainPrep.csv'
alphabetPath = "alphabet.txt"
lenAlpha = len(open(alphabetPath).read())

# divide IDs in valid and train IDs
# TODO: super in elegant
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
model.add(Conv1D(filters=256, kernel_size=7, activation="relu", padding="same", input_shape=(maxChars, lenAlpha)))
model.add(MaxPooling1D(3))

model.add(Conv1D(256, 7, activation="relu", padding="same"))

model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(256, 3, activation="relu", padding="same"))

model.add(Conv1D(256, 3, activation="relu", padding="same"))

model.add(Conv1D(256, 3, activation="relu", padding="same"))

model.add(Conv1D(256, 3, activation="relu", padding="same"))

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

plot_model(model, to_file="archcharCnn8Huge.png", show_layer_names=True, show_shapes=True)

############ train

# TODO: try binary_crossentropy and change buildSetDG in pl accordingly

# Define optimizer for the model
# adam = optimizers.Adam(lr = 0.0001,
#                        beta_1 = 0.9,
#                        beta_2 = 0.999,
#                        epsilon = 1e-08,
#                        decay = 1e-08)

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
                                           patience = 5,
                                           verbose = 0,
                                           mode = 'auto')

csvLogger = keras.callbacks.CSVLogger("charCnn8HugeLog.csv")

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

model.save("charCnn8Huge.h5")

# architecture plot
plot_model(model, to_file="charCnn8Huge"
                          ".png", show_layer_names=True, show_shapes=True)


print("\n and done with saving")

