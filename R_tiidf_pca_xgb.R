# test if wrod2vec will help the imputation
library(xgboost)
nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition_clean_data.csv")
pca <- prcomp(nutri1, center = TRUE, scale. = TRUE) 
plot(pca, type = "l")
pca$x[,1:10]
nutri = cbind(nutri, pca$x[,1:10])

main_sub = c("CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45')


CEREAL <- nutri[nutri$SubDeptCode==main_sub[1]]
CEREAL = na.omit(CEREAL)

model1_mse = mean((CEREAL$`Sugars per 100g`-mean(CEREAL$`Sugars per 100g`))^2)

a = dist(CEREAL[,21:30])
a = as.matrix(a)
b = colSums(a)
c = t(a)/b
d = rowSums(c*CEREAL$`Sugars per 100g`)

model2_mse = mean((CEREAL$`Sugars per 100g`-d)^2)
#################################################
#test = 
x_train= fread("/Users/xiaofeifei/I/Oxford/Dissertation/train_1.csv",header = T)
x_test = fread("/Users/xiaofeifei/I/Oxford/Dissertation/test_1.csv",header = T)
y_train = read.csv("/Users/xiaofeifei/I/Oxford/Dissertation/train_y.csv",header = T)
y_test = read.csv("/Users/xiaofeifei/I/Oxford/Dissertation/test_y.csv",header = T)

pca <- prcomp(x_train, center = TRUE, scale. = TRUE) 
plot(pca, type = "l")
train = pca$x[,1:10]
test = predict(pca, x_test)[,1:10]

xgb <- xgboost(data = data.matrix(train), 
               label = y_train$Sugars.per.100g, 
               eta = 0.1,
               max_depth = 15, 
               nround=250, 
               eval_metric = "rmse",
               objective = "reg:linear",
               nthread = 6,
               base_score = mean(y_train$Sugars.per.100g),
               early_stopping_rounds = 10
)

model1_mse = mean((y_test$Sugars.per.100g-mean(y_train$Sugars.per.100g))^2)
model2_mse = mean((y_test$Sugars.per.100g-predict(xgb,test))^2)

# try test micro department
CEREAL_train = CEREAL[51:325,]
CEREAL_train[,mean := mean(`Sugars per 100g`), by = MicroDeptCode]
CEREAL_train =CEREAL_train[!duplicated(CEREAL_train$MicroDeptCode), c("MicroDeptCode", "mean")]

CEREAL_test = CEREAL[1:50,]

setkey(CEREAL_train, MicroDeptCode)
setkey(CEREAL_test, MicroDeptCode)

CEREAL_test = merge(CEREAL_test, CEREAL_train, all.x = T)
model1_mse = mean((CEREAL_test$`Sugars per 100g` - CEREAL_test$mean)^2)
