# First, you should analyse the computational (time) complexity of solving the penalised
# least squares problem for a given (λ1, λ2)

# O(N^2)

library(penalized)
library(doParallel)
data("nki70")
attach(nki70)

lambda1_final = numeric()
lambda2_final = numeric()
cor = numeric()
#Input: 
lambda1 = 10
lambda2 = 10
stepping1 = 1
stepping2 = 1
epsilon = 0.001

####################### DONT FORGET NORMALIZING ################
########
####
##
#
# 1. 10 runs of 10-fold cross validation of lambda1 and lambda2

# inital 6 cores for parallel computing 
cl <- makeCluster(6)
registerDoParallel(cl)

# inital ten runs
trials <- 10

r <- foreach(icount(trials), .combine=cbind, .packages=c('penalized')) %dopar% {
  
  # 10-fold cross validation
  fit <- cvl(nki70$time, nki70[, 8:77], lambda1 = lambda1, lambda2 =lambda2, fold = 10,
             model = "linear")
  # predictive correlations
  cor(nki70$time, fit$predictions[,1])
  
}
stopCluster(cl)

# average of the 10 predictive correlations
base_pred_cor = abs(mean(r))

# 2. (a) “neighbours" search 
i = 0
while(TRUE)
{
  
  i = i+1
  lambda11 = append(lambda1, c(lambda1+stepping1, lambda1-stepping1))
  lambda22 = append(lambda2, c(lambda2+stepping2, lambda2-stepping2))
  
  all_models = expand.grid(lambda11,lambda22)
  
  cl <- makeCluster(6)
  registerDoParallel(cl)
  
  trials <- 10
  r = foreach (k = 1:dim(all_models)[1]) %:%
  foreach(icount(trials), .combine=cbind, .packages=c('penalized')) %dopar% {
    
    # 10-fold cross validation
    fit <- cvl(nki70$time, nki70[, 8:77], lambda1 = all_models[k,1], lambda2 =all_models[k,2], fold = 10,
               model = "linear")
    # predictive correlations
    cor(nki70$time, fit$predictions[,1])
    
  }
  stopCluster(cl)
  
  predict_cor = abs(sapply(r, mean))
  
  if(any(predict_cor-base_pred_cor>epsilon))
  {
    lambda1 = as.numeric(all_models[which.max(predict_cor),][1,1])
    lambda2 = as.numeric(all_models[which.max(predict_cor),][1,2])
    base_pred_cor = max(predict_cor)
  }else{
    lambda1 = lambda1
    lambda2 = lambda2
    break
  }
  
  if(lambda1 < stepping1)
  {
    stepping1 = lambda1/2
  }
  
  if(lambda2 < stepping2)
  {
    stepping2 = lambda2/2
  }
}

lambda1_final = append(lambda1_final, lambda1)
lambda2_final = append(lambda2_final, lambda2)
cor = append(cor, base_pred_cor)
