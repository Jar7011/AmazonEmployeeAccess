# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)

## BART Model ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
bart_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 4)

# Create random forest model
bart_model <- bart(trees = 1000) %>% 
  set_engine('dbarts') %>% 
  set_mode('classification')

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Set workflow
bart_wf <- workflow() %>% 
  add_recipe(bart_recipe) %>% 
  add_model(bart_model) %>% 
  fit(data = train_data)

# Predict
bart_preds <- predict(bart_wf, 
                      new_data = test_data,
                      type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = bart_preds, file = "./BART.csv", delim = ",")

## Score: 0.83024
