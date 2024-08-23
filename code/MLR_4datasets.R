library(nnet)
library(purrr)
library(caret)
require(reshape2)
library(smotefamily)
# # read data and set data type----------
# data_complete <- read.csv("data/data_complete.csv",  header = TRUE)
# data_mean <- read.csv("data/data_mean.csv",  header = TRUE)[, -1]
# data_ml <- read.csv("data/data_ml.csv",  header = TRUE)[, -1]
data_mice <- read.csv("data/data_mice.csv",  header = TRUE,)
str(data_mice)


df_list <- list(data_complete, data_mean,data_ml,data_mice)
#The map function from the purrr package in R is used to apply a function to each element of a list or a vector

reset_column_types <- function(df) {
  logical_columns <- readLines("data/logical_columns.txt")
  factor_columns <- readLines("data/factor_columns.txt")
  
  df[logical_columns] <- lapply(df[logical_columns], as.logical)
  df[factor_columns] <- lapply(df[factor_columns], as.factor)
  
  return(df)
}


df_list <- map(df_list, reset_column_types)

# delete  urgency=schedule-----------
del_schedule <- function(df)
  
  
  
  
data_complete <- subset(df_list[[1]], Csection.Urgency != "Schedule")
data_complete$Csection.Urgency <- droplevels(data_complete$Csection.Urgency[data_complete$Csection.Urgency != "Schedule"])
data_mean <- subset(df_list[[2]], Csection.Urgency != "Schedule")
data_mean$Csection.Urgency <- droplevels(data_mean$Csection.Urgency[data_mean$Csection.Urgency != "Schedule"])
data_ml <- subset(df_list[[3]], Csection.Urgency != "Schedule")
data_ml$Csection.Urgency <- droplevels(data_ml$Csection.Urgency[data_ml$Csection.Urgency != "Schedule"])
data_mice <- subset(df_list[[4]], Csection.Urgency != "Schedule")
data_mice$Csection.Urgency <- droplevels(data_mice$Csection.Urgency[data_mice$Csection.Urgency != "Schedule"])

# delete 2 variables that are almost same with response
data_complete <- data_complete %>% select(-Method.of.Delivery.A,-Csection.Incidence)
data_mean <- data_mean %>% select(-Method.of.Delivery.A,-Csection.Incidence)
data_ml <- data_ml %>% select(-Method.of.Delivery.A,-Csection.Incidence)
data_mice <- data_mice %>% select(-Method.of.Delivery.A,-Csection.Incidence)

str(data_mean)
str(data_ml)

# data partition------------
# Split using caret
set.seed(42)  # For reproducibility
trainIndex_com <- createDataPartition(data_complete$Csection.Urgency, p = 0.7, list = FALSE)
train_data_com <- data_complete[trainIndex_com, ]
test_data_com <- data_complete[-trainIndex_com, ]

trainIndex_23 <- createDataPartition(data_mean$Csection.Urgency, p = 0.7, list = FALSE)
train_data_mean <- data_mean[trainIndex_23, ]
test_data_mean <- data_mean[-trainIndex_23, ]
train_data_ml <- data_ml[trainIndex_23, ]
test_data_ml <- data_ml[-trainIndex_23, ]
trainIndex_4 <- createDataPartition(data_mice$Csection.Urgency, p = 0.7, list = FALSE)
train_data_mice <- data_mice[trainIndex_4, ]
test_data_mice <- data_mice[-trainIndex_4, ]

# MLR-----------------
# complete
str(train_data_com)
train_data_com$Csection.Urgency2 <- relevel(train_data_com$Csection.Urgency, ref = "Nonemer")
train_data_com <- train_data_com %>% select(-Csection.Urgency)
model_complete <- multinom(Csection.Urgency2~., data = train_data_com)
# Summary of the model
summary(model_complete)
## extract the coefficients from the model and exponentiate
exp(coef(model_complete))
#predicted probabilities 
model_res_com0 <- predict(model_complete, newdata = test_data_com, "probs")
model_res_com1 <- predict(model_complete, newdata = test_data_com, "class")
v <- data.frame(test_data_com$Csection.Urgency)
model_res_com <- cbind(data.frame(model_res_com1),v)
colnames(model_res_com) <- c('Predict','Actual')
# contigency table
table(model_res_com$Predict,model_res_com$Actual)
#accuracy
sum(model_res_com$Predict == model_res_com$Actual)/nrow(model_res_com)
# [1] 0.9893617






# mean
#str(train_data_mean)
train_data_mean$Csection.Urgency2 <- relevel(train_data_mean$Csection.Urgency, ref = "Nonemer")
train_data_mean <- train_data_mean %>% select(-Csection.Urgency)
model_mean <- multinom(Csection.Urgency2~., data = train_data_mean)
# Summary of the model
summary(model_mean)
## extract the coefficients from the model and exponentiate
exp(coef(model_mean))
#predicted probabilities 
model_res_mean0 <- predict(model_mean, newdata = test_data_mean, "probs")
model_res_mean1 <- predict(model_mean, newdata = test_data_mean, "class")
v2 <- data.frame(test_data_mean$Csection.Urgency)
model_res_mean <- cbind(data.frame(model_res_mean1),v2)
colnames(model_res_mean) <- c('Predict','Actual')

# contigency table
table(model_res_mean$Predict,model_res_mean$Actual)
#accuracy
sum(model_res_mean$Predict == model_res_mean$Actual)/nrow(model_res_mean)#[1] 0.9458992




# ml------
# Error in nnet.default(X, Y, w, mask = mask, size = 0, skip = TRUE, softmax = TRUE,  : 
#                         too many (1050) weights
str(train_data_ml)
train_data_ml$Csection.Urgency2 <- relevel(train_data_ml$Csection.Urgency, ref = "Nonemer")
train_data_ml <- train_data_ml %>% select(-Csection.Urgency)
model_ml <- multinom(Csection.Urgency2~., data = train_data_ml)
# Summary of the model
summary(model_ml)
## extract the coefficients from the model and exponentiate
exp(coef(model_ml))
#predicted probabilities 
model_res_ml0 <- predict(model_ml, newdata = test_data_ml, "probs")
model_res_ml1 <- predict(model_ml, newdata = test_data_ml, "class")
v3 <- data.frame(test_data_ml$Csection.Urgency)
model_res_ml <- cbind(data.frame(model_res_ml1),v3)
colnames(model_res_ml) <- c('Predict','Actual')

# contigency table
table(model_res_ml$Predict,model_res_ml$Actual)
#accuracy
sum(model_res_ml$Predict == model_res_ml$Actual)/nrow(model_res_ml)#[1] 0.9465484


# mice
#str(train_data_mean)
train_data_mice$Csection.Urgency2 <- relevel(train_data_mice$Csection.Urgency, ref = "Nonemer")
train_data_mice <- train_data_mice %>% select(-Csection.Urgency)
model_mice <- multinom(Csection.Urgency2~., data = train_data_mice)
# Summary of the model
summary(model_mice)
## extract the coefficients from the model and exponentiate
exp(coef(model_mice))
#predicted probabilities 
model_res_mice0 <- predict(model_mice, newdata = test_data_mice, "probs")
model_res_mice1 <- predict(model_mice, newdata = test_data_mice, "class")
v4 <- data.frame(test_data_mice$Csection.Urgency)
model_res_mice <- cbind(data.frame(model_res_mice1),v4)
colnames(model_res_mice) <- c('Predict','Actual')
model_res_mice <- na.omit(model_res_mice)
# contigency table
table(model_res_mice$Predict,model_res_mice$Actual)
#accuracy
sum(model_res_mice$Predict == model_res_mice$Actual)/nrow(model_res_mice)#[1] 0.94722

