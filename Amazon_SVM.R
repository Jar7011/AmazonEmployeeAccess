# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)
library(embed)

## Support Vector Machine Model ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
svm_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)

# Create random forest model
svm_model <- svm_rbf(rbf_sigma = tune(), 
                     cost = tune()) %>% 
  set_engine('kernlab') %>% 
  set_mode('classification')

# Set workflow
svm_wf <- workflow() %>% 
  add_recipe(svm_recipe) %>% 
  add_model(svm_model)

# Grid of values to tune over
svm_grid_params <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5)

# Split data for cross validation
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
detectCores()
cl <- makePSOCKcluster(10)
registerDoParallel(10)

# Run the cv
cv_results <- svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_grid_params,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow and fit it
final_wf <- svm_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = train_data)

# Predict and format for submission
svm_preds <- predict(final_wf, 
                     new_data = test_data,
                     type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = svm_preds, file = "./Support_Vector_Machine.csv", delim = ",")

# Score with PCA: 0.77339