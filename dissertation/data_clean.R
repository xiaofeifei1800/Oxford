library(data.table)
weights = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition weights data.csv")
sales = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition sales data.csv")

# combine data
setkey(weights, SkuCode)
setkey(sales, SkuCode)
nutri = merge(sales,weights[,SkuName:=NULL],all.x = T)
rm(sales, weights)

########################
# clean data
########################
# change to numeric
str(nutri)
changeCols = c("SkuCode", "Energy per portion", "Sugars per Portion", "Fat per Portion",
               "Saturates per Portion", "Salt per Portion")
nutri[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]
str(nutri)

# check NA
# could use NLP to impute missing values
apply(is.na(nutri),2,sum)
by(nutri, nutri_100g$OwnBrand, function(x) 
  apply(is.na(x),2,sum))


# portion of NA by deptcode
by(nutri, nutri_100g$DeptCode, function(x) 
  apply(is.na(x),2,sum))

# check duplicate
sum(duplicated(nutri))

# remode duplicate
nutri = nutri[!duplicated(nutri),]

# check value consistencies
summary(nutri)
# no negative values

# check extremely values
boxplot(nutri[,11:12], main = "Boxplot of variables")
na.omit(nutri[nutri$`Energy per portion` >=900,])
# child meal
#
boxplot(nutri[,13:20], main = "Boxplot of variables", cex=0.7,axes=FALSE)
axis(2,cex.axis=1) 
axis(1,1:8,colnames(nutri[,13:20]),cex.axis=0.5,las=2)
# fat per 100g
na.omit(nutri[nutri$`Fat per 100g` >=50,])
# nuts

# fat per portion
na.omit(nutri[nutri$`Fat per Portion` >=50,])
# wired

# Saturates per 100g
na.omit(nutri[nutri$`Saturates per 100g` >=50,])
# Butter

# Saturates per portion
na.omit(nutri[nutri$`Saturates per Portion`>=40,])
# ice cream?

# Suger per 100g
na.omit(nutri[nutri$`Sugars per 100g`>=50,])
# candy or cake

# Suger per portion
na.omit(nutri[nutri$`Sugars per Portion`>=40,])
# candy or cake

# Salt per 100g
na.omit(nutri[nutri$`Salt per 100g`>=10,])
# No idea

fwrite(nutri, file = "/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition data.csv")
