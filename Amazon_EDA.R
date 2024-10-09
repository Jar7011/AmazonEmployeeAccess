library(vroom)
library(embed)
library(tidymodels)
library(DataExplorer)


# Read in the data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Simple Exploratory Plots
plot_intro(train_data)
plot_missing(train_data)
plot_correlation(train_data)


# Create a recipe
my_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001, other = 'Other') %>% 
  step_dummy(all_nominal_predictors())

# Prep and bake recipe
prep <- prep(my_recipe)  
baked <- bake(prep, new_data = train_data)

# Get number of columns
ncol(baked)
