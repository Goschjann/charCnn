# Parse json data file on restaurant reviews for the text classification task



library(jsonlite)
library(dplyr)
library(Matrix)
library(parallel)
library(foreach)

#############################
#
#         Get Data from yelp github repo
#
#############################


url = "https://raw.githubusercontent.com/rekiksab/Yelp/master/yelp_challenge/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json"


# 230 K observations
zips = jsonlite::stream_in(url(url))

#######################
#
#   create data samples
#
#######################

# binary classification / sentiment analysis
# creates data frame with text and corresponding sentiment for each of the observations
# can also be adjusted to 5-class classification task for the stars
# 4,5 are good, 1, 2, 3, are bad reviews. Better balanced data
# lower case the characters!

# Still super unbalanced dataset
# TODO

sentData = zips %>%
  select(stars, text) %>%
  # filter(stars != 3) %>%
  mutate(sentiment = ifelse(stars < 4, 0, 1)) %>%
  mutate(textLow = casefold(text, upper = FALSE)) %>%
  select(textLow, sentiment) %>%
  # kill all reviews without content
  filter(textLow != "")
# check
table(sentData$sentiment)
table(zips$stars)

object.size(sentData)

write.csv(sentData, file = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv", row.names = FALSE)
checkWrite = read.csv("/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv", header = TRUE)

# train and test split
set.seed(1337)
trainratio = 3/4
trainIndices = sample(1:nrow(sentData), size = ceiling(trainratio*nrow(sentData)),
  replace = FALSE)
train = sentData[ trainIndices, ]
test = sentData[ -trainIndices, ]

# more or less equally balanced
table(train$sentiment) / sum(table(train$sentiment))
table(test$sentiment) / sum(table(test$sentiment))

write.csv(train, file = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTrain.csv", row.names = FALSE)
write.csv(test, file = "/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTest.csv", row.names = FALSE)

checkWrite = read.csv("/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentDataTest.csv", header = TRUE)



########################
#
#   PG
#
#######################

# empty text reviews?
train$nchar = nchar(train$textLow)
head(train)

which(train$nchar == 0)




# textlow


a = test$textLow[1]
nchar(a)


a = "Affe"
b = casefold(a, upper = FALSE)
a
b

# transform the texts to data matrices with maximal 1014 characters
# each character is represented as a vector of length length(alphabet)
# and for each text we receive a data matrix with dimension 70x1014

# read in alphabet, this alphabet is shorter, only 53 symbols
alphabet = as.character(read.table(file = '/home/jgucci/Desktop/uni/text_mining/tm_data/alphabet')$V1)
charAlphabet = strsplit(x = alphabet, split = "")[[1]]


########################
#
#       Transform to sparse data matrix
#
#####################

# input: one test + sentiment
# output of this function: one large vector with "flattened" character-representation
# matrix + sentiment

text_to_crep = function(text, maxWords, charAlphabet) {
  # convert text
  text = strsplit(text, split = "")[[1]]
  # counter
  i = 1
  # instantiate repMatrix
  repMatrix = matrix(0, ncol = maxWords, nrow = length(charAlphabet))
  finalMatrix = matrix(0, nrow = )

  # only enter maxWords entries
  for (char in text) {
    #print(char)
    #print(which(charAlphabet %in% char))
    if (i > maxWords) {
      break
    }
    if (i <= length(text)) {
      repMatrix[which(charAlphabet %in% char), i] = 1
    }
    i = i + 1
  }
  # output vector
  return(as.vector(repMatrix))
}

## apply to all 230K observations
# takes 3hr, super stupid solution approach

maxWords = 1014
sentDataSelection = sentData
dataMatrixSparse = Matrix::sparseMatrix(i = 1, j = 1, x = 1, dims = c(nrow(sentDataSelection), ((maxWords*length(charAlphabet) + 1))))
# start = proc.time()
for (j in 1:nrow(sentDataSelection)) {
  # j = 2
  dataMatrixSparse[j, ] = c(text_to_crep(text = sentDataSelection[j, "text"], maxWords = 1014,
    charAlphabet = charAlphabet), sentDataSelection[j, "sentiment"])
  if (j %% 1000 == 0) {
    print(paste0("processing ", j))
  }
}
# stop = proc.time()
dim(dataMatrixSparse)
object.size(dataMatrixSparse)
# print(stop - start)

dataMatrixSparse[25000:25500, 53743]

#### Problem: R cannot handle matrixes with that many entries
#### siwtch to python for that ;-/

Matrix::writeMM(dataMatrixSparse, "/home/jgucci/Desktop/uni/text_mining/tm_data/charnn_input_big.mtx")



#################
#################

# testdf = as.data.frame(rbind(c("jann", 1), c("max", 2), c("iris", 3)))
# colnames(testdf) = c("text", "sentiment")
# testdf$text = as.character(testdf$text)
# testdf$sentiment = as.numeric(as.character(testdf$sentiment))
# str(testdf)
# testdf
#
# maxWords = 1014
# dataMatrix = Matrix::sparseMatrix(i = 1, j = 1, x = 1, dims = c(nrow(testdf), ((maxWords*length(charAlphabet) + 1))))
#
# start = proc.time()
# for (j in 1:nrow(testdf)) {
#   # j = 2
#   dataMatrix[j, ] = c(text_to_crep(text = testdf[j, "text"], maxWords = 1014,
#     charAlphabet = charAlphabet), testdf[j, "sentiment"])
#   if (j %% 10000 == 0) {
#     print(paste0("processing ", j))
#   }
# }
#
# dim(dataMatrix)
# dataMatrix[, 53743]
#
# dataMatrixSparse = as(object = dataMatrix, Class = "sparseMatrix")


########### TRIAL with lappply and mclapply for parallelization
text_to_crep_sparse = function(text, maxWords, charAlphabet) {
  # convert text
  text = strsplit(text, split = "")[[1]]
  # counter
  i = 1
  # instantiate repMatrix
  repMatrix = matrix(0, ncol = maxWords, nrow = length(charAlphabet))
  finalMatrix = matrix(0, nrow = )

  # only enter maxWords entries
  for (char in text) {
    #print(char)
    #print(which(charAlphabet %in% char))
    if (i > maxWords) {
      break
    }
    if (i <= length(text)) {
      repMatrix[which(charAlphabet %in% char), i] = 1
    }
    i = i + 1
  }
  # output vector
  return(as(as.vector(repMatrix), "sparseVector"))
}




testdf = as.data.frame(rbind(c("jann", 1), c("max", 2), c("iris", 3)))
colnames(testdf) = c("text", "sentiment")
testdf$text = as.character(testdf$text)
testdf$sentiment = as.numeric(as.character(testdf$sentiment))
str(testdf)
testdf

write.csv(testdf, file = "/home/jgucci/Desktop/uni/text_mining/tm_data/test.csv", row.names = FALSE)





foo = t(sapply(testdf$text, function(x) text_to_crep_sparse(x, maxWords = 1014, charAlphabet = charAlphabet)))

fooMat = lapply(foo, as, "sparseMatrix")
fooMat = do.call(rBind, fooMat)
dim(fooMat)
fooMat[, 1:10]


