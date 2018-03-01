"""
Library for CharCNN Project
Jann Goschenhofer
Feb 20. 2018
"""

import scipy.io
import pandas
import numpy as np
import collections
import csv
import random
from keras.utils.np_utils import to_categorical


"""
Classic check function
"""

def check():
    print("projectlib loaded HUI")

"""
convert text to matrix of character strings

input: text as string, alphabet as string, amount of chars per observation (paper: 1014)
output: ndarray with dimension (#alphabet, #maxchars, 1), aka a matrix 
"""

def generate_one_hot(text, alphabet, maxChars):

    #  initialize empty ndarray with zeros only and depth 1
    textRep = np.zeros(shape=(len(alphabet), maxChars, 1))

    # cut text to maxChars
    if len(text) > maxChars:
        text = text[0:maxChars]

    # loop over all chars in the text
    for char_index in range(0, len(text)):
        if text[char_index] in alphabet:
            alpha_index = alphabet.find(text[char_index])
            # rows = alphabet features, columns = characters
            textRep[alpha_index][char_index][0] = 1
        # in case of whitespace or unknown characters include 0-vector
        # do nothing

    return(textRep)


##########
##########      Decode one hot encoding
##########

def decoder(onehotText, alphabet, maxChars = 1014):
    # initialize string
    # onehotText = encoded
    a = str()
    for colIndex in range(0, maxChars):
        # only store alphabet index, if 1 is in column
        if np.isin(1, onehotText[:, colIndex, 0]):
            alphaIndex = np.where(onehotText[:, colIndex, 0] == 1)[0][0]
            # check print
            # print(colIndex, alphaIndex)
            a = a + alphabet[alphaIndex]
        # else add blank spacre
        else:
            a = a + " "
    return(a)

##########
##########      Decode one hot encoding for 2dim objects
##########
#TODO: integrate in above function!

def decoder2dim(onehotText, alphabet, maxChars = 1014):
    # initialize string
    # onehotText = encoded
    a = str()
    for colIndex in range(0, maxChars):
        # only store alphabet index, if 1 is in column
        if np.isin(1, onehotText[:, colIndex]):
            alphaIndex = np.where(onehotText[:, colIndex] == 1)[0][0]
            # check print
            # print(colIndex, alphaIndex)
            a = a + alphabet[alphaIndex]
        # else add blank spacre
        else:
            a = a + " "
    return(a)


##############
############## Encode user written review life
##############

def encodeReview(textInput, path_alphabet, maxChars):

    data = pandas.DataFrame(pandas.Series(textInput))

    alphabet = open(path_alphabet).read()

    # generate data Matrix
    #reviews = data.iloc[0:amountData, 0]
    reviewRep = np.zeros(shape=(1, (maxChars * len(alphabet)), 1))
    # stor.shape

    for i in range(len(reviews)):
        reviewRep[i, :, 0] = generate_one_hot(text=reviews[i],
                                         alphabet=alphabet,
                                         maxChars=maxChars).reshape(-1, len(alphabet) * maxChars, order="F")
    # generate label vector
    labels = data.iloc[0:amountData, 1]

    return(reviewRep[:, :, 0], labels)




##############
############## Function that returns matrix representation and sentiment for x reviews
##############

def buildSet(path_data, path_alphabet, maxChars, amountData, batchStart = 0):

    if batchStart != 0:
        data = pandas.concat(pandas.read_csv(filepath_or_buffer=path_data, skiprows=batchStart, chunksize=amountData),
                             ignore_index=True)
    else:
        data = pandas.read_csv(filepath_or_buffer=path_data)

    alphabet = open(path_alphabet).read()

    # generate data Matrix
    reviews = data.iloc[0:amountData, 0]
    reviewRep = np.zeros(shape=(len(reviews), (maxChars * len(alphabet)), 1))
    # stor.shape

    for i in range(len(reviews)):
        reviewRep[i, :, 0] = generate_one_hot(text=reviews[i],
                                         alphabet=alphabet,
                                         maxChars=maxChars).reshape(-1, len(alphabet) * maxChars, order="F")
    # generate label vector
    labels = data.iloc[0:amountData, 1]

    return(reviewRep[:, :, 0], labels)



############## BUILD TEST DATA SET
############## Function that returns matrix representation and sentiment for x reviews FROM already read array
##############

# needed in the Data Generator
# Difference: outputs data in reshaped shape and readable for keras
# amount data is not needed anymore
# uses skiprows to skip the reading of some rows

def buildSetTest(path_data, path_alphabet, maxChars, skiprows = 0, amountData = 1000):

    data = pandas.read_csv(filepath_or_buffer=path_data, skiprows=skiprows)


    alphabet = open(path_alphabet).read()
    lenAlpha = len(alphabet)

    # generate data Matrix#
    # amount data to work with
    reviews = data.iloc[0:amountData, 0]
    #print(len(reviews))
    reviewRep = np.zeros(shape=(len(reviews), (maxChars * len(alphabet)), 1))
    # stor.shape

    for i in range(len(reviews)):
        reviewRep[i, :, 0] = generate_one_hot(text=reviews[i],
                                         alphabet=alphabet,
                                         maxChars=maxChars).reshape(-1, len(alphabet) * maxChars, order="F")
    # generate label vector
    # hacky but ok
    # TODO check error with dimensions!!
    labels = data.iloc[0:amountData, 1]#





    # return(reviewRep[:, :, 0].reshape(-1, lenAlpha, maxChars, 1), labels)
    return(reviewRep[:, :, 0].reshape(-1, maxChars, lenAlpha), labels)




##############
############## Function that returns matrix representation and sentiment for x reviews FROM already read array
##############

# needed in the Data Generator
# Difference: outputs data in reshaped shape
# amount data is not needed anymore
# uses skiprows to skip the reading of some rows

def buildSetDG(path_data, path_alphabet, maxChars, skiprows = 0):

    data = pandas.read_csv(filepath_or_buffer=path_data, skiprows=skiprows)


    alphabet = open(path_alphabet).read()
    lenAlpha = len(alphabet)

    # generate data Matrix
    reviews = data.iloc[:, 0]
    reviewRep = np.zeros(shape=(len(reviews), (maxChars * len(alphabet)), 1))
    # stor.shape

    for i in range(len(reviews)):
        reviewRep[i, :, 0] = generate_one_hot(text=reviews[i],
                                         alphabet=alphabet,
                                         maxChars=maxChars).reshape(-1, len(alphabet) * maxChars, order="F")
    # generate label vector
    labels = to_categorical(data.iloc[:, 1])


    # changed for 2 net
    return(reviewRep[:, :, 0].reshape(-1, maxChars, lenAlpha), labels)




##############
############## Data Generator Object Class
##############

# excellent blog post: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# adjusted a lot to pre-build buildSetDG


# we need shape (trainSize, lenAlpha, maxChars, 1) for input X
# dim_x = lenAlpha, dim_y = maxChars, batch_size = batch_size
# we do not have a dim z as we use 1-dimensional images

import numpy as np

class DataGenerator(object):
    # intitialize the whole vehicle
    def __init__(self, dim_x = 32, dim_y = 32, batch_size = 32, shuffle = True, path_data = "bla", path_alphabet = "bla",
                 maxChars = 1014, allIDs = range(0, 2)):
          'Initialization'
          self.dim_x = dim_x
          self.dim_y = dim_y
          self.batch_size = batch_size
          self.shuffle = shuffle
          self.path_data = path_data
          self.path_alphabet = path_alphabet
          self.maxChars = maxChars
          self.allIDs = allIDs

    # again: we do not need labels as opposed to the blog post
    # listIDs are given to that homie during the model.fit_generator() call
    def generate(self, list_IDs):
          'Generates batches of samples'
          # Infinite loop
          while 1:
              # Generate order of exploration of dataset
              # to add randomness
              indexes = self.__get_exploration_order(list_IDs)

              # Generate batches
              imax = int(len(indexes)/self.batch_size)
              for i in range(imax):
                  # Find list of IDs
                  list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                  # Generate data
                  X, y = self.__data_generation(list_IDs_temp)

                  yield X, y

    # changes order of indices in the batches if shuffle is set true
    # add randomness to data
    def __get_exploration_order(self, list_IDs):
          'Generates order of exploration'
          # Find exploration order
          indexes = np.arange(len(list_IDs))
          if self.shuffle == True:
              np.random.shuffle(indexes)

          return indexes

    # generates batches
    # only need IDs
    # IDs are beforehand shuffled by __get_exploration_order()
    # we do not need the labels, automatically extracted by DG build
    def __data_generation(self, list_IDs_temp):
          'Generates data of batch_size samples'

          # Generate data
          # that is the FUNKY PART
          # read only the lines from the review thingy that match the current list_IDs_temp
          # aka skip all lines that do not match
          # use np.setdiff1d to find the skip-IDs

          deselectIDs = np.setdiff1d(self.allIDs, list_IDs_temp)

          # check printer
          # print("I start generating a data set")

          X, y = buildSetDG(path_data=self.path_data, path_alphabet=self.path_alphabet,
                                        maxChars=self.maxChars, skiprows=deselectIDs)

          # return X, sparsify(y)
          return X, y






