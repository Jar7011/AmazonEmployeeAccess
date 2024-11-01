library(tidymodels)
library(embed)
library(doParallel)
library(vroom)
library(themis)

## Penalized Logistic Regression ## 

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
penalized_log_reg_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 4)

# Create model
penalized_log_reg_model <- logistic_reg(mixture = tune(),
                                        penalty = tune()) %>% 
  set_engine('glmnet')

# Create workflow
penalized_log_reg_wf <- workflow() %>% 
  add_recipe(penalized_log_reg_recipe) %>% 
  add_model(penalized_log_reg_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
detectCores() # 8
cl <- makePSOCKcluster(8)
registerDoParallel(8)

# Run the CV
cv_results <- penalized_log_reg_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow
final_wf <- penalized_log_reg_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
log_reg_preds <- predict(final_wf, 
                         new_data = test_data,
                         type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = log_reg_preds, file = "./Penalized_Log_Reg.csv", delim = ",")

# Score: 0.78337
# Score with PCA: 0.77701
# Score with SMOTE algorithm: 0.78343