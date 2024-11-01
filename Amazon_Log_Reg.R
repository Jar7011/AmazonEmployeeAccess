library(vroom)
library(tidymodels)
library(tidyverse)
library(doParallel)
library(themis)

## Logistic Regression ##

# Read in the data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
log_reg_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 4)

# Create logistic regression model
log_reg_model <- logistic_reg() %>% 
  set_engine('glm')

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Create workflow
log_reg_wf <- workflow() %>% 
  add_recipe(log_reg_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data = train_data)

# Predict and format for submission
log_reg_preds <- predict(log_reg_wf, 
                         new_data = test_data,
                         type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = log_reg_preds, file = "./Logistic_Reg.csv", delim = ",")

# Score: 0.80913
# Score with PCA: 0.79650
# Score with SMOTE algorithm: 0.80214
