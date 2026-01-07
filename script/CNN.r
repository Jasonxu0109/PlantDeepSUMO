library(Biostrings)
library(keras)
library(tensorflow)
library(ggplot2)
library(ROCR) 
library(pROC)
library(ggsci)
library(cowplot)
library(reshape2)
library(patchwork)
library(yardstick)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(precrec)
library(tidymodels)
library(themis)
library(plotrix)
EncodingSeq <- function(sequence){
    A <- "100000000000000000000"
    C <- "010000000000000000000"
    D <- "001000000000000000000"
    E <- "000100000000000000000"
    F <- "000010000000000000000"
    G <- "000001000000000000000"
    H <- "000000100000000000000"
    I <- "000000010000000000000"
    K <- "000000001000000000000"
    L <- "000000000100000000000"
    M <- "000000000010000000000"
    N <- "000000000001000000000"
    P <- "000000000000100000000"
    Q <- "000000000000010000000"
    R <- "000000000000001000000"
    S <- "000000000000000100000"
    T <- "000000000000000010000"
    V <- "000000000000000001000"
    W <- "000000000000000000100"
    X <- "000000000000000000010"
    Y <- "000000000000000000001"
    encoded_seq <- lapply(sequence,function(x){
        x <- toupper(x)
        x <- gsub("A",A,x)
        x <- gsub("C",C,x)
        x <- gsub("D",D,x)
        x <- gsub("E",E,x)
        x <- gsub("F",F,x)
        x <- gsub("G",G,x)
        x <- gsub("H",H,x)
        x <- gsub("I",I,x)
        x <- gsub("K",K,x)
        x <- gsub("L",L,x)
        x <- gsub("M",M,x)
        x <- gsub("N",N,x)
        x <- gsub("P",P,x)
        x <- gsub("Q",Q,x)
        x <- gsub("R",R,x)
        x <- gsub("S",S,x)
        x <- gsub("T",T,x)
        x <- gsub("V",V,x)
        x <- gsub("W",W,x)
        x <- gsub("X",X,x)
        x <- gsub("Y",Y,x)
    })
    encoded_seq <- unlist(encoded_seq)
    return(encoded_seq)
}
convStringToMatrix <- function(encodedSeqs, seq_len=30){
  # ensure the character type of encodedSeqs
  encodedSeqs <- as.character(encodedSeqs)
  # create the feature matrix:
  x_array <- array(data = 0, dim = c(21,seq_len, length(encodedSeqs)))
  s <- 1 # sequence/instance index
  r <- 1 # row of the matrix, each row represents A,T/U, G, C
  c <- 1 # column of the matrix, each column represents each nucleotide in the 100nt sequence
  j <- 1 # index of character in the one-hot encoded string
  # store each character into the right place of 3D matrix
  while (s <= length(encodedSeqs)) {
    c <- 1
    while (c <= seq_len) {
      r <- 1
      while (r <= 21) {
        x_array[r,c,s] <- as.integer(substr(encodedSeqs[s], j,j))
        r <- r + 1
        j <- j + 1
      }
      c <- c + 1
    }
    s <- s + 1
    j <- 1
  }

  #change the index order of x_array to the one keras package required:
  x_array_new <- aperm(x_array,c(3,2,1))
  return(x_array_new)
}
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
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)
# ----------------------------
# 参数化模型构建函数
# ----------------------------
build_model <- function(filters = 64, kernel_size = 5) {
    tf$random$set_seed(123)
    set.seed(123)
    set_random_seed(123)
  model <- keras_model_sequential() %>%
    # 第一卷积层
    layer_conv_1d(filters = filters, kernel_size = kernel_size, padding = "valid",
                 activation = "relu", input_shape = c(60, 21)) %>%
    layer_dropout(0.5)%>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_dropout(0.5)%>%
    layer_flatten() %>%
    # 输出层
    layer_dense(units = 2, activation = "softmax")
  # 编译模型
        model %>% compile(
        optimizer = optimizer_adam(learning_rate = 0.001),
        loss = "binary_crossentropy",
        metrics = c("accuracy", 
                   tf$keras$metrics$AUC(name = "auc")))
  return(model)
}
# 定义参数组合
hyper_grid <- expand.grid(
  filters = c(32, 64, 128),
  kernel_size = c(3, 5, 7),
  batch_size = c(32, 64)
)
train <- read.csv("model/train_positive_negative_balance.csv")
train$encoded_seq <- EncodingSeq(train$Sequence)
train$label <- ifelse(train$Label=='negative',0,1)
test <- read.csv("model/test_positive_negative_balance.csv")
test$encoded_seq <- EncodingSeq(test$Sequence)
test$label <- ifelse(test$Label=='negative',0,1)
flank_len <- 30
x_train <- convStringToMatrix(train$encoded_seq,seq_len=flank_len*2) 
y_train <- keras3::to_categorical(train$label,2) 
x_test <- convStringToMatrix(test$encoded_seq,seq_len=flank_len*2) 
y_test <- keras3::to_categorical(test$label,2) 
# 存储结果
results <- data.frame()
for(i in 1:nrow(hyper_grid)) {
  # 模型配置
  model <- build_model(
    filters = hyper_grid$filters[i],
    kernel_size = hyper_grid$kernel_size[i]
  )
  # 训练模型
  history <- model %>% fit(
    x_train, y_train,
    batch_size = hyper_grid$batch_size[i],
    epochs = 10,
    validation_split = 0.1,
    verbose = 0,
    callbacks = list(early_stop)
  )
  # 记录结果
  best_epoch <- which.min(history$metrics$val_loss)
  results <- rbind(results, data.frame(
    filters = hyper_grid$filters[i],
    kernel_size = hyper_grid$kernel_size[i],
    batch_size = hyper_grid$batch_size[i],
    val_auc = history$metrics$val_auc[best_epoch],
    val_acc = history$metrics$val_acc[best_epoch]
  ))
}
# 找出最佳参数组合
best_params <- results[which.max(results$val_auc),]
best_params
result <- NULL
for(cv in c(10)){
    k <- cv  # 10折交叉验证
    set.seed(123)
    folds <- createFolds(train$label, k = k) 
    # 存储每折的准确率
    accuracies <- numeric(k)
      # 划分训练集和验证集
    for (i in 1:length(folds)){
        cat("Processing fold", i, "\n")
      # 分割数据
      testIndex <- folds[[i]]
      train_x <- x_train[-testIndex,1:60,1:21]
      train_y <- y_train[-testIndex,]
      val_x <- x_train[testIndex,1:60,1:21]
      val_y <- y_train[testIndex,]
      # 训练模型
      final_model <- build_model(
          filters = best_params$filters,
          kernel_size = best_params$kernel_size
      )
      history <- final_model %>% fit(
        train_x, train_y,
        batch_size = best_params$batch_size,
        epochs = 10,
        validation_data = list(val_x, val_y),
        callbacks = list(
          callback_early_stopping(patience = 10),
          callback_reduce_lr_on_plateau(factor = 0.5, patience = 3)
        )
      )
      # 计算验证集准确率
      #results <- model %>% evaluate(val_x, val_y, verbose = 0)
        pred <- final_model %>% predict(val_x)
        pred <- pred[,2]
        re <- metrics(pred,val_y[,2])
        re <- data.frame(Type='train',re)
        pred <- final_model %>% predict(x_test)
        pred <- pred[,2]
        re2 <- metrics(pred,y_test[,2])
        re2 <- data.frame(Type='test',re2)
        result <- rbind(result,re,re2)
    }
}
result %>%
  reshape2::melt(id=c('Type','cv','fold')) %>%
  group_by(Type,cv,variable) %>%
  summarise(Frequency=mean(value),se=std.error(value)) %>%
  as.data.frame() %>%
  arrange(Type,variable)





























