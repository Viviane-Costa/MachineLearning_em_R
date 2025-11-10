## Introdução ao Machine Learning com o R
## Pacote tidymodels
## Viviane Costa Silva - UFLA

## Vários modelos ao mesmo tempo

## Carregando o pacote

library(tidymodels)

## Banco de dados 

data(ames)


# divisão dos dados
splits_ame <- ames %>%
  initial_split(strata = Sale_Price) 

ames_train <- training(splits_ame)
ames_test  <- testing(splits_ame)

## Pré - Processamento dos dados (recipe)
recipe_simples <- recipe(
  Sale_Price ~ Gr_Liv_Area + Longitude + Latitude,
  data = ames_train
) 

## Modelos

dt_model <- decision_tree(
  cost_complexity = tune(), min_n = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

rf_model <- rand_forest(
  mtry = tune(), min_n = tune(), trees = 1000
) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

xgb_model <- boost_tree(
  tree_depth = tune(), learn_rate = tune(), 
  loss_reduction = tune(),  min_n = tune(), 
  sample_size = tune(), trees = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

##  workflow
work <- workflow_set(
  preproc = list(simple = recipe_simples), 
  models = list(decision_tree = dt_model,
                rand_forest = rf_model, 
                xgboost = xgb_model
  ))


## Treinando
cv_splits <- vfold_cv(
  ames_train, v = 10, strata = Sale_Price
)

grid_ctrl <- control_grid(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)

grid_results <- work  %>%
  workflow_map(
    resamples = cv_splits,
    grid = 15,
    control = grid_ctrl
  )

autoplot(grid_results)

autoplot(
  grid_results,
  rank_metric = "rmse",  
  metric = "rmse",       
  select_best = TRUE     
)

autoplot(
  grid_results,
  rank_metric = "rsq",  
  metric = "rsq",       
  select_best = TRUE  
)


save.image("Varios_modelosss.RData")

