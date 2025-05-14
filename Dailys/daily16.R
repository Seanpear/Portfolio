---
title: "Daily Exercise 15"
author: "Sean Pearson"
date: "2025-04-06"
format: html
execute:
  echo: true
---

# Load necessary packages
library(tidymodels)
library(palmerpenguins)

# Set seed for reproducibility
set.seed(123)

# Drop rows with missing values
penguins_clean <- penguins %>% drop_na()

# Split the data into training (70%) and testing (30%)
penguin_split <- initial_split(penguins_clean, prop = 0.7)
penguin_train <- training(penguin_split)
penguin_test  <- testing(penguin_split)

# Create 10-fold cross-validation set from training data
penguin_folds <- vfold_cv(penguin_train, v = 10)

# Part 2: Daily Exercise 16

# Define logistic regression model
log_mod <- logistic_reg(mode = "classification")

# Define random forest model (default engine is "ranger")
rf_mod <- rand_forest(mode = "classification")

# Create simple recipe for modeling: predict species from bill_length_mm and flipper_length_mm
penguin_recipe <- recipe(species ~ bill_length_mm + flipper_length_mm, data = penguin_train)

# Set up workflow set
penguin_models <- workflow_set(
  preproc = list(simple = penguin_recipe),
  models = list(logistic = log_mod, random_forest = rf_mod)
)

# Fit models using cross-validation
model_results <- penguin_models %>%
  workflow_map(resamples = penguin_folds, metrics = metric_set(accuracy))

# Show ranked results
rank_results(model_results)

#The random forest model is best because it got higher accuracy (0.953) than the logistic regression model (0.597).
