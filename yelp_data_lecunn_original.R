# yann le cun train and test data original
library(dplyr)

train = read.csv(file = "..data/train.csv", header = FALSE)
test = read.csv(file = "..data/test.csv", header = FALSE)

# set to lower case
# labels to 0 (= positive) and 1 (= negative)
trainPrep = train %>%
  mutate(textLow = casefold(V2, upper = FALSE)) %>%
  mutate(sentiment = ifelse(V1 == 1, 1, 0)) %>%
  select(textLow, sentiment)

testPrep = test %>%
  mutate(textLow = casefold(V2, upper = FALSE)) %>%
  mutate(sentiment = ifelse(V1 == 1, 1, 0)) %>%
  select(textLow, sentiment)

write.csv(file = "..data/trainPrep.csv", x = trainPrep,
  row.names = FALSE)
write.csv(file = "..data/testPrep.csv", x = trainPrep,
  row.names = FALSE)

# also store randomly sampled samller subsets
size = 0.5
trainIDs = sample(1:nrow(train), size = ceiling(nrow(train)*size), replace = FALSE)
testIDs = sample(1:nrow(test), size = ceiling(nrow(test)*size), replace = FALSE)

trainPrepSmall = trainPrep[ trainIDs, ]
testPrepSmall = testPrep[ testIDs, ]
# balanced?
table(testPrepSmall$sentiment)
table(trainPrepSmall$sentiment)

write.csv(file = "..data/trainPrepSmall.csv", x = trainPrepSmall,
  row.names = FALSE)
write.csv(file = "..data/testPrepSmall.csv", x = testPrepSmall,
  row.names = FALSE)
