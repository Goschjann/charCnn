import scipy.io
import pandas
import numpy as np
import math
import keras
import random
import collections
import csv

import projectlib as pl

# read sentData exported from R
dataSent = pandas.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv")

dataSent1 = pandas.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv",
                            skiprows=trainIDs)



# memory usage in bytes
dataSent.memory_usage()
# roughly 3 GB of RAM




# select with iloc

### onehot encoding for text
alphabet = open("/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt").read()


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


##############
############## Function , that returns matrix representation and sentiment for x reviews
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


revCheck, labCheck = buildSet(path_data="/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv",
                 path_alphabet="/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt",
                 maxChars=1014,
                 amountData=20)


revCheck1, labCheck1 = buildSet(path_data="/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv",
                 path_alphabet="/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet.txt",
                 maxChars=1014,
                 amountData=10, batchStart= 10)

test1 = pandas.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_data/test.csv")
test2 = pandas.concat(pandas.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_data/test.csv",
                        skiprows=2, chunksize=1), ignore_index=True)


foo1 = generate_one_hot(text=test1, alphabet= alphabet, maxChars=8)




#################
################# Data generator
#################


data = pandas.read_csv(filepath_or_buffer="/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv",
                                     skiprows=range(0, 100000))



data = pandas.concat(pandas.read_csv(filepath_or_buffer=path_data, skiprows=batchStart, chunksize=amountData),
                             ignore_index=True)


# excellent blog post: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

# divide IDs in valid and train IDs
x = 10
x = len(dataSent)
allIDs = range(0, x)



trainIDs = random.sample(allIDs, math.ceil(0.2 * x))
validIDs = np.setdiff1d(allIDs, a)

# dictionary that stores IDs for train and validation
partition = {"train": trainIDs, "validation": validIDs}
# dictionary that stores matching labels
labels = {allIDs: dataSent.iloc[:, 1]}


# we need shape (trainSize, lenAlpha, maxChars, 1) for input X
# dim_x = lenAlpha, dim_y = maxChars, batch_size = batch_size
# we do not have a dim z as we use 1-dimensional images

import numpy as np

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, dim_x = 32, dim_y = 32, batch_size = 32, shuffle = True, path_data, path_alphabet, maxChars):
          'Initialization'
          self.dim_x = dim_x
          self.dim_y = dim_y
          self.batch_size = batch_size
          self.shuffle = shuffle
          self.path_data = path_data
          self.path_alphabet = path_alphabet
          self.maxChars = maxChars

    # again: do not need labels
    # list ideas are given to that homie during the model.fit_generator() call
    def generate(self, list_IDs):
          'Generates batches of samples'
          # Infinite loop
          while 1:
              # Generate order of exploration of dataset
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
    def __get_exploration_order(self, list_IDs):
          'Generates order of exploration'
          # Find exploration order
          indexes = np.arange(len(list_IDs))
          if self.shuffle == True:
              np.random.shuffle(indexes)

          return indexes
    # generates batches
    # only need IDs and labels
    # IDs are beforehand shuffled by __get_exploration_order()
    # we do not need the labels, automatically extracted by DG build
    def __data_generation(self, list_IDs_temp):
          'Generates data of batch_size samples'

          # Generate data
          # that is the FUNKY PART
          # read only the lines from the review thingy that match the current list_IDs_temp
          # aka skip all lines that do not match
          # use np.setdiff1d to find the skip-IDs

          deselectIDs = np.setdiff1d(allIDs, list_IDs_temp)

          X, y = pl.buildSetDG(path_data=self.path_data, path_alphabet=self.path_alphabet,
                                        maxChars=self.maxChars, skiprows=deselectIDs)

          # return X, sparsify(y)
          return X, y

# maybe include later
#TODO: not needed because of 2-class problem

def sparsify(y):
  'Returns labels in binary NumPy array'
  n_classes = # Enter number of classes
  return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])






#############
############# Playground
#############


maxChars = 5
testData = ["affe", "bffe", "cffe"]
stor = np.zeros(shape=(len(testData), (maxChars*len(alphabet)), 1))
stor.shape

for i in range(len(testData)):
    stor[i, :, 0] = generate_one_hot(text= testData[i],
                                     alphabet= alphabet,
                                     maxChars = maxChars).reshape(-1, len(alphabet) * maxChars, order = "F")
check = stor[:, :, 0]


a = np.zeros(shape=(3, 4, 1))
a.shape
b = a[:, :, 0]

c = np.array([[1,2,3]])
c
c.shape
a[0, :, 0] = np.array([1,2,3,4])







testText = "bffe"

maxChars = 5
foo = generate_one_hot(text = testText, alphabet = alphabet, maxChars=maxChars)[:, :, 0]
foors = foo.reshape(-1, len(alphabet)*maxChars, order = "F")

foo.shape
# works:
foo[0, 0, 0]
foo[5, 1, 0]
collections.Counter(foo[:, 4, 0])

## toy around with reshape
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = a.reshape(-1, 9)
c = a.reshape(-1)

d = foo.reshape(-1, len(alphabet) * maxChars)
d.shape





##### apply on 5 texts from SubSentData

subSentData = dataSent.iloc[0:5,0]
subSentData[1]

subSentData.shape
maxChars = 1014



foo = dataSent.iloc()





#### goal: numpy matrix with dimension [5, maxChars*len(alphabet)]
# initialize empty numpy array with this dimensions

stor = np.zeros(shape=(len(subSentData), (maxChars*len(alphabet)), 1))
stor.shape

for i in range(len(subSentData)):
    stor[i, :, 0] = generate_one_hot(text= subSentData[i],
                                     alphabet= alphabet,
                                     maxChars = maxChars).reshape(-1, len(alphabet) * maxChars, order = "F")
check = stor[:, :, 0]





subSentData = dataSent.iloc[0:5,0]
print(subSentData[0:10])
len(subSentData)

1hot = keras.preprocessing.text(text = subSentData,
                                )
maxChars = 1014