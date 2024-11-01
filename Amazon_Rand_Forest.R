# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)
library(embed)
library(themis)

## Random Forest Model ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
rand_forest_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 3)

# Create random forest model
rand_forest_model <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 1000) %>% 
  set_engine('ranger') %>% 
  set_mode('classification')

# Set workflow
rand_forest_wf <- workflow() %>% 
  add_recipe(rand_forest_recipe) %>% 
  add_model(rand_forest_model)

# Grid of values to tune over
prep(rand_forest_recipe) %>% bake(train_data) %>% ncol() # = 10
rand_forest_grid_params <- grid_regular(mtry(range = c(1, 10)),
                                        min_n(),
                                        levels = 5)

# Split data for cross validation
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(10)
registerDoParallel(10)

# Run the cv
cv_results <- rand_forest_wf %>% 
  tune_grid(resamples = folds,
            grid = rand_forest_grid_params,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow and fit it
final_wf <- rand_forest_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = train_data)

# Predict and format for submission
rand_forest_preds <- predict(final_wf, 
                     new_data = test_data,
                     type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = rand_forest_preds, file = "./Random_Forest.csv", delim = ",")

# Score: 0.87473
# Score with PCA: 0.84959
# Score with SMOTE algorithm: 0.86158