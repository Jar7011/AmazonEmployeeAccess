library(vroom)
library(embed)
library(tidymodels)

# Read in the data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

