rm(list=ls())
library(ggplot2)
library(ROCR) 
library(pROC)
library(cowplot)
library(reshape2)
library(patchwork)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(precrec)
library(tidymodels)
library(themis)
library(stringr)
library(lightgbm)
library(doParallel)
library(plotrix)
library(ranger)
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
setwd("/home/data/t0202029/chaoqun/PlantDeepSUMO")
load("model//train_test_dataset.RData")

x_train <- matrix(x_train, nrow = dim(x_train)[1], ncol = dim(x_train)[2] * dim(x_train)[3])
x_train <- data.frame(x_train)
x_train$label <- train$label

x_test <- matrix(x_test, nrow = dim(x_test)[1], ncol = dim(x_test)[2] * dim(x_test)[3])
x_test <- data.frame(x_test)
x_test$label <- test$label
metrics <- function(Pred,realY){
  AUC_ROC <- roc(realY,Pred)
  AUC_ROC <- AUC_ROC$auc
  pred.class <- as.integer(Pred > 0.5)
  TBL <- table(pred.class, realY)
  TP <- TBL[2,2]
  FP <- TBL[2,1]                  
  TN <- TBL[1,1]                                                                     
  FN <- TBL[1,2]
  Specificity <- TN/(TN+FP)
  Sensitivity <- TP/(TP+FN)
  Precision <- TP/(TP+FP)
  Recall <- TP/(TP+FN)
  F1 <- 2 * (Precision*Recall/(Precision+Recall))
  MCC <- (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN))/sqrt((TN+FP)*(TN+FN))
  Accuracy <- (TP+TN)/(TP+TN+FP+FN)
  tmp <- data.frame(V1=realY, V2=Pred)
  tmp$V1 <- factor(tmp$V1)
  tmp <- evalmod(scores = tmp$V2, labels = tmp$V1)
  aucs <- auc(tmp)
  AUC_PRC <- subset(aucs, curvetypes == "PRC")
  AUC_PRC <- AUC_PRC$aucs
  re <- data.frame(cv=cv,fold=i,Sensitivity,Specificity,Precision,Recall,Accuracy,AUC_ROC,AUC_PRC,MCC,F1)
  return(re)
}
result <- NULL
for(cv in 10){
  k <- cv  # 10折交叉验证
  set.seed(123)
  folds <- createFolds(train$label, k = k) 
  # 存储每折的准确率
  accuracies <- numeric(k)
  # 划分训练集和验证集
  for (i in 1:length(folds)){
    cat("Processing fold", i, "\n")
    # 划分训练集和验证集
    testIndex <- folds[[i]]
    train_x <- x_train[-testIndex,]
    train_y <- x_train$label[-testIndex]
    val_x <- x_train[testIndex,]
    val_y <- x_train$label[testIndex]
    # 构建和训练模型
    rf_model <- ranger(
      label ~ ., 
      data = train_x,  
      num.trees = 500,
      mtry = 2, 
      importance = "impurity" 
    )
    # 计算验证集准确率
    #results <- model %>% evaluate(val_x, val_y, verbose = 0)
    pred <- predict(rf_model, val_x)
    pred <- pred$predictions
    re <- metrics(pred,val_y)
    re <- data.frame(Type='train',re)
    pred <- predict(rf_model, x_test)
    pred <- pred$predictions
    re2 <- metrics(pred,x_test$label)
    re2 <- data.frame(Type='test',re2)
    result <- rbind(result,re,re2)
  }
}
write.csv(result,"model/evaluation/rf_train_evaluation.csv",row.names=F)
#######
result %>%
  reshape2::melt(id=c('Type','cv','fold')) %>%
  group_by(Type,cv,variable) %>%
  summarise(Frequency=mean(value),se=std.error(value)) %>%
  as.data.frame() %>%
  arrange(Type,variable)

