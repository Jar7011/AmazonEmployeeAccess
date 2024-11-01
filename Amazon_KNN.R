library(tidymodels)
library(doParallel)
library(vroom)

## KNN ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change ACTION to factors
train_data <- train_data %>% 
  mutate(ACTION = as.factor(ACTION))

# Create a recipe
knn_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(all_outcomes(), neighbors = 4)

# Create knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('kknn')

# Create workflow
knn_wf <- workflow() %>% 
  add_recipe(knn_recipe) %>% 
  add_model(knn_model)

# Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 10)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Run the CV
cv_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow
final_wf <- knn_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
knn_preds <- predict(final_wf, 
                         new_data = test_data,
                         type = "prob") %>% 
  mutate(id = row_number(), ACTION = .pred_1) %>% 
  select(id, ACTION)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = knn_preds, file = "./KNN.csv", delim = ",")

# Score: 0.80913
# Score with PCA: 0.80058
# Score with SMOTE algorithm: 0.80224