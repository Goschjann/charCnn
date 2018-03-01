# yann le cun train and test data original
library(dplyr)

train = read.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/yelp_review_polarity_csv/train.csv", header = FALSE)
test = read.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/yelp_review_polarity_csv/test.csv", header = FALSE)

# set to lower case
# labels to 0 (= negative) and 1 (= positive)
trainPrep = train %>%
  mutate(textLow = casefold(V2, upper = FALSE)) %>%
  mutate(sentiment = ifelse(V1 == 1, 1, 0)) %>%
  # kill all reviews without content
  filter(textLow != "") %>%
  select(textLow, sentiment)

testPrep = test %>%
  mutate(textLow = casefold(V2, upper = FALSE)) %>%
  mutate(sentiment = ifelse(V1 == 1, 1, 0)) %>%
  # kill all reviews without content
  filter(textLow != "") %>%
  select(textLow, sentiment)

write.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/trainPrep.csv", x = trainPrep,
  row.names = FALSE)
write.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/testPrep.csv", x = trainPrep,
  row.names = FALSE)

# also store randomly sampled samller subsets
size = 0.4
trainIDs = sample(1:nrow(train), size = ceiling(nrow(train)*size), replace = FALSE)
testIDs = sample(1:nrow(test), size = ceiling(nrow(test)*size), replace = FALSE)

trainPrepSmall = trainPrep[ trainIDs, ]
testPrepSmall = testPrep[ testIDs, ]
# balanced?
table(testPrepSmall$sentiment)
table(trainPrepSmall$sentiment)

write.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/trainPrepSmall.csv", x = trainPrepSmall,
  row.names = FALSE)
write.csv(file = "Desktop/uni/text_mining/tm_data/yelp_polarity/testPrepSmall.csv", x = testPrepSmall,
  row.names = FALSE)


checkWrite = read.csv("/home/jgucci/Desktop/uni/text_mining/tm_data/yelp_sentData.csv", header = TRUE)

a = trainPrepSmall[1, "textLow"]
b = checkWrite[1, "textLow"]
a
b

class(b)
class(a)


str(checkWrite)
str(test)
