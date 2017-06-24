library(sm)
library(data.table)

nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition data.csv")
nutri1 = fread("/Users/xiaofeifei/I/Oxford/Dissertation/Nutrition text.csv", header = T)
# SUB 5 only have 6obs, len = 14
main_sub = c("CEREAL", 'YOG', '311', '310', 'XBISC', 'LAY013', 'LAY083', 'DAY-SWEETS',
             'ICREAM', '335', '338', '390', '396', '45')
  
# k mean example
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- nutri[nutri$SubDeptCode==main_sub[1]]
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

sm.density.compare(data$`Sugars per 100g`, as.factor(data$MicroDeptCode))

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

# project back
mu = colMeans(data[,11:20, with=F])
X = pca$x[,1:nComp] %*% t(pca$rotation[,1:nComp])
X = scale(X, center = -mu, scale = FALSE)
X = as.data.table(X)

sm.density.compare(X$`Sugars per 100g`, cluster$cluster)
sm.density.compare(X$`Sugars per Portion`, cluster$cluster)

k1 = numeric()
k2 = numeric()
p = numeric()

k1 = append(k1,0)
k2 = append(k2,2)
p = append(p,2)

# k1 = 2 0 4 2 0 0 0 0 3 3 0 0 2 0
# k2 = 0 3 3 2 0 0 0 2 3 3 2 0 2 2
# p = 5 5 4 4 0 6 4 4 3 5 4 4 4 2



