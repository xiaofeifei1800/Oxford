nutri = fread("/Users/xiaofeifei/I/Oxford/Dissertation/k_mean_data.csv")
colnames(nutri) = as.character(nutri[1,])
nutri = nutri[-1,]

# k mean example
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){kmeans(nutri[,1:4, with=F], k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

cluster = kmeans(nutri[,1:4, with=F], 14, nstart=50,iter.max = 15 )

sm.density.compare(as.numeric(nutri$Sugars), cluster$cluster)

boxplot(as.numeric(nutri$Sugars)~cluster$cluster, main="Boxplot of sugar in new groups", 
        xlab="New groups", ylab="Sugar value")












