---
title: "Daily Exercise 9"
author: "Sean Pearson"
date: "2025-04-07"
format: html
execute:
  echo: true
---

# Load packages
library(tidyverse)
library(tidymodels)
library(janitor)
library(ggplot2)

# Load data
url <- 'https://raw.githubusercontent.com/mikejohnson51/csu-ess-330/refs/heads/main/resources/co-est2023-alldata.csv'
data <- read_csv(url) |>
  clean_names() |>
  filter(county == "000") |>       # Keep only state-level data
  filter(!is.na(deaths2020))       # Drop rows missing target variable

# Select features for modeling
data_model <- data |>
  select(
    state_name = stname,
    popestimate2020,
    births2020,
    netmig2020,
    naturalchg2020,
    gqestimates2020,
    deaths2020
  )

# Reduce skew
data_model <- data_model |>
  mutate(log_deaths = log(deaths2020 + 1))

# Split data into training and test sets
set.seed(42)
split <- initial_split(data_model, prop = 0.8, strata = log_deaths)
train <- training(split)
test <- testing(split)

# Cross-validation folds
folds <- vfold_cv(train, v = 5)

# Recipe
rec <- recipe(log_deaths ~ ., data = train) |>
  update_role(state_name, new_role = "id") |>
  step_normalize(all_numeric_predictors())

# Model: Random Forest
rf_model <- rand_forest(mode = "regression", trees = 500) |>
  set_engine("ranger")

# Workflow
wf <- workflow() |>
  add_recipe(rec) |>
  add_model(rf_model)

# Resampling
fit_resampled <- fit_resamples(wf, resamples = folds, metrics = metric_set(rmse, rsq))

# Final model fit
final_fit <- fit(wf, data = train)

# Predict on test set
preds <- predict(final_fit, new_data = test) |>
  bind_cols(test)

# Evaluate
metrics(preds, truth = log_deaths, estimate = .pred)

# Plot actual vs predicted
ggplot(preds, aes(x = log_deaths, y = .pred)) +
  geom_point() +
  geom_abline(color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Log(Deaths 2020)",
    x = "Actual Log(Deaths)",
    y = "Predicted Log(Deaths)"
  ) +
  theme_minimal()

# Save plot
ggsave("truth_vs_predicted.png", width = 6, height = 4, dpi = 300)
