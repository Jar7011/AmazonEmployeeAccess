# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)
library(embed)
library(discrim)

## Naive Bayes Model ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
nb_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

# Create random forest model
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_engine('naivebayes') %>% 
  set_mode('classification')

# Set workflow
nb_wf <- workflow() %>% 
  add_recipe(nb_recipe) %>% 
  add_model(nb_model)

# Grid of values to tune over
nb_grid_params <- grid_regular(Laplace(),
                                        smoothness(),
                                        levels = 5)

# Split data for cross validation
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Run the cv
cv_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_grid_params,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow and fit it
final_wf <- nb_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = train_data)

# Predict and format for submission
nb_preds <- predict(final_wf, 
                    new_data = test_data,
                    type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = nb_preds, file = "./Naive_Bayes.csv", delim = ",")

# Score: 0.75864
