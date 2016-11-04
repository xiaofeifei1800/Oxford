# 1 load data
loblolly = read.table("P:/R Data/prepd-loblolly.txt.xz", header = T)
# check intger
sapply(loblolly, function(x) class(x))

# take a look at data
str(loblolly)
summary(loblolly)
sapply(loblolly, function(x)any(is.na(x)))

#2
library(penalized)
ridge = penalized(T, penalized = loblolly[,2:2000],lambda2 = 60, data = loblolly, model = "linear")

#3
library(lattice)
str(ridge)
xyplot(loblolly$T ~ ridge@fitted, main = "fitted against T", xlab = "Fitted value", ylab = "T value",
       panel = function(x, y) 
         {
          panel.xyplot(x, y)
          panel.abline(a = 0 , b = 1, col="red")
          panel.abline(lm(y ~ x), col = "grey")
         })

#4
library(latticeExtra)
xyplot(ridge@fitted ~ ridge@residuals, main = "fitted against Residual", xlab = "Fitted value", ylab = "Residual value",
       panel = function(x, y) 
       {
         panel.xyplot(x, y)
         panel.abline(a = 1, col="grey")
         panel.loess(x,y)
       })

#5
xval = function(data, fold)
{
  g = factor(rep(1:fold,ceiling(nrow(data)/10)))[1:859]
  cv_data = split(data, g)
  NewVar <- function(x) {  
    train = cv_data[-x]
    train = do.call(rbind,train)
    test = cv_data[[x]]
    ridge = penalized(T, penalized = train[,2:2000],lambda2 = 60, data = train, model = "linear")
    pred_value = predict(ridge, test[,2:2000],data = test)
    cbind(pred_value[,1],test[,1])
  }
  pair = lapply(1:flod, function(x) NewVar(x))
  pair = do.call(rbind, pair)
  corr = cor(pair[,1],pair[,2])
  return(corr)
}













