## Introdução ao Machine Learning com o R
## Pacote tidymodels
## Viviane Costa Silva - UFLA


## Regressão no tidymodels 

## Carregando o pacote

library(tidymodels)

## Banco de Dados 

Dados_Iris <- iris[-c(1:20), ] # Retirei as 20 primeiras linhas do banco de dados 
Dados_Iris[c(1, 51, 101), 1] <- NA # Valores ausentes 
Dados_Iris[c(10, 60, 110), 5] <- NA # Valores ausentes 
Dados_Iris <- dplyr::tibble(Dados_Iris) # converte o conjunto de dados em um objeto tibble 

## Descritiva 

glimpse(Dados_Iris)
summary(Dados_Iris)
names(Dados_Iris)

## Gráficos 

# Gráfico de barras da contagem de espécies
ggplot(Dados_Iris, aes(x = Species)) +
  geom_bar() +
  labs(title = "Gráfico de Barras da Contagem de Espécies", x = "Espécie", y = "Contagem")

# Gráfico de dispersão de Sepal.Length vs. Sepal.Width
ggplot(Dados_Iris, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point(aes(color = Species)) +
  labs(title = "Gráfico de Dispersão Sepal.Length vs. Sepal.Width", x = "Comprimento da Sépala", y = "Largura da Sépala")

# Boxplot das medidas das pétalas para cada espécie
ggplot(Dados_Iris, aes(x = Species, y = Petal.Length)) +
  geom_boxplot() +
  labs(title = "Boxplot das Medidas das Pétalas por Espécie", x = "Espécie", y = "Comprimento da Pétala")


## Divisão dos dados (splits)

Divisão_Iris <- initial_split(Dados_Iris, prop = 0.7, strata = Species)
Divisão_Iris

iris_treinamento <- training(Divisão_Iris) 
iris_teste <- testing(Divisão_Iris)


## Algoritmo de regressão linear (engine)

reg_mod <- linear_reg(
  penalty = tune(), mixture = tune()
) %>% 
  set_engine("glmnet")

# A função tune() executa uma busca sistemática na grade de hiperparâmetros, 
# treinando e avaliando o modelo para cada combinação de hiperparâmetros.


## Pré - Processamento dos dados (recipe)

reg_recipe <- recipe(
  Petal.Length ~ ., data = iris_treinamento
) %>% 
  step_impute_knn(all_predictors()) %>% # lida com valores ausentes nos dados
  step_dummy(all_nominal_predictors()) %>% # lida com variáveis categóricas (nominais) convertendo-as em variáveis binárias 
  step_normalize(all_numeric_predictors()) # normalização

## workflow

reg_workflow <- workflow() %>% 
  add_model(reg_mod) %>% 
  add_recipe(reg_recipe)

## Validação Cruzada

val_set <- vfold_cv(
  iris_treinamento, v = 4, strata = Species
)

## Treinamento 

reg_trained <- reg_workflow %>% 
  tune_grid(
    val_set,
    grid = 10,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(rmse,mae, rsq)
  )


## 10 melhores modelos 

reg_trained %>% show_best(n = 10, "mae")


## Vizualização desses modelos 

# autoplot
ggplot2::autoplot(reg_trained)

## Seleção do melhor modelo, segundo a métrica estabelecida 

reg_best_tune <- select_best(reg_trained, "rmse")
final_reg_model <- reg_mod %>% 
  finalize_model(reg_best_tune)

## Teste

workflow() %>% 
  add_recipe(reg_recipe) %>% 
  add_model(final_reg_model) %>% 
  last_fit(Divisão_Iris) %>% 
  collect_predictions() %>% 
  select(Petal.Length, .pred) %>% 
  ggplot() +
  aes(x= Petal.Length, y = .pred) +
  geom_point()

## Vizualinado as métricas para o teste

workflow() %>% 
  add_recipe(reg_recipe) %>% 
  add_model(final_reg_model) %>% 
  last_fit(Divisão_Iris) %>% 
  collect_metrics()
