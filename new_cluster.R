library(data.table)
library(sm)
nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/k_mean_data.csv")
ori = fread("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
nutri = nutri[-1,]
nutri = nutri[,-1]
nutri = nutri[,101:124]
nutri[,V125:=NULL]
nutri[is.na(nutri)] = 0

changeCols = colnames(nutri)
nutri[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]

is.special <- function(x){
  if (is.numeric(x)) !is.finite(x) else is.na(x)
}
colSums(sapply(data, is.special))

nutri = scale(nutri)
nutri = as.data.table(nutri)

nutri[is.na(nutri)] = 0
# k mean example
nutri[,SubDeptCode := ori$SubDeptCode]
nutri[,Sugars := ori$Sugars]
set.seed(123)

plot(density(nutri$Sugars))

cluster = kmeans(nutri$Sugars, 2, nstart=50,iter.max = 15 )

sm.density.compare(nutri$Sugars, cluster$cluster)

nutri[,class:= cluster$cluster]
ori[,class:= cluster$cluster]
fwrite(ori, "/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")

# cluster by each sub

main_sub = c("CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45')

data <- nutri[nutri$SubDeptCode==main_sub[4]]

plot(density(data$Sugars))


###

library(sm)
library(data.table)

nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/sub_g_data.csv")
nutri = na.omit(nutri)
# SUB 5 only have 6obs, len = 14
main_sub = c("CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45')


# k mean example
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- nutri[nutri$SubDeptCode%in% c("311", "310")]
data = na.omit(data)
wss <- sapply(1:k.max, 
              function(k){kmeans(data[,11:20, with=F], k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

cluster = kmeans(data[,11:20, with=F], 2, nstart=50,iter.max = 15 )

sm.density.compare(data$`Sugars per 100g`, cluster$cluster)

micro_clu1 = unique(data$SubDeptCode[cluster$cluster == 1])
micro_clu2 = unique(data$SubDeptCode[cluster$cluster == 2])

intersect(micro_clu1,micro_clu2)

# pca
pca <- prcomp(data[,11:20, with=F], center = TRUE, scale. = TRUE) 
plot(pca, type = "l")
biplot(pca, scale = 0)

# project to PC
nComp = 2
pca.data = pca$x[,1:nComp]

# performe cluster again
wss <- sapply(1:k.max, 
              function(k){kmeans(pca.data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

cluster = kmeans(pca.data, 2, nstart=50,iter.max = 15 )

mu = colMeans(data[,11:20, with=F])
X = pca$x[,1:nComp] %*% t(pca$rotation[,1:nComp])
X = scale(X, center = -mu, scale = FALSE)
X = as.data.table(X)

sm.density.compare(X$`Sugars per 100g`, cluster$cluster)

micro_clu1 = unique(data$MicroDeptCode[cluster$cluster == 1])
micro_clu2 = unique(data$MicroDeptCode[cluster$cluster == 2])

intersect(micro_clu1,micro_clu2)

###########
nutri = nutri[order(SubDeptCode)]
data <- nutri[nutri$SubDeptCode%in% main_sub]
plot(data$Sugars, col = factor(data$SubDeptCode))

# combine 311 and 310

nutri$SubDeptCode[nutri$SubDeptCode == "311"] = "310"

data <- nutri[nutri$SubDeptCode%in% c("ICREAM", "DAY-SWEETS")]
plot(data$Sugars, col = factor(data$SubDeptCode))

nutri$SubDeptCode[nutri$SubDeptCode == "ICREAM"] = "DAY-SWEETS"

fwrite(nutri, file = "/Users/xiaofeifei/I/Oxford/Dissertation/merge_data.csv", row.names = F)


















