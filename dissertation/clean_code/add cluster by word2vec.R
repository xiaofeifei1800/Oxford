library(data.table)
library(dplyr)
library(ggplot2)
word2vec = fread("/Users/xiaofeifei/I/Oxford/Dissertation/train_feature.csv", header = T)

# pca
pca <- prcomp(word2vec[,1:100, with=F], center = TRUE, scale. = TRUE)
x = cumsum(pca$sdev)
normalized = (x-min(x))/(max(x)-min(x))

normalized = as.data.frame(normalized)

ggplot(aes(y = normalized, x = 1:100), data= normalized) + 
  geom_point(colour = "red") + 
  geom_line(stat="identity",colour = "red") +
  xlab("Number of principle components") + ylab("Percentage of variance")+
  theme(axis.text=element_text(size=15),
        axis.title.y = element_text(size=20),
        axis.title.x = element_text(size=20))

plot(normalized, type = "l", main = "Total variance explained by number of principle components", xlab = "Number of principle components",
     ylab = "Percentage of variance")

# choose 21

nComp = 21
pca.data = pca$x[,1:nComp]

# performe cluster again
size <- sapply(1:13, 
              function(k){kmeans(pca.data, k, nstart=50,iter.max = 15 )$size})
wss <- sapply(1:13, 
              function(k){kmeans(pca.data, k, nstart=50,iter.max = 15 )$tot.within})

mse = mapply("/",wss,size,SIMPLIFY = FALSE)
mse = lapply(mse, function(x) mean(x))

mse = as.data.frame(do.call(rbind, mse))
wss = as.data.frame(wss)
ggplot(aes(y = wss/93, x = 1:13), data= wss) + 
  geom_point(colour = "red") + 
  geom_line(stat="identity",colour = "red") +
  xlab("Number of clusters K") + ylab("Average within-clusters mean squared error") +
  theme(axis.text=element_text(size=15),
        axis.title.y = element_text(size=20),
        axis.title.x = element_text(size=20))

plot(1:13, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main = "Total within-clusters sum of squares plot")

cluster = kmeans(pca.data, 14, nstart=50,iter.max = 50 )
word2vec$cluster = cluster$cluster

new_group = word2vec %>%
  group_by(cluster)%>%
  summarise(variance = var(Sugars))


old_group = word2vec %>%
  group_by(SubDeptName)%>%
  summarise(variance = var(Sugars))

################# try for micro
sub = unique(word2vec$SubDeptName)
new_data = data.table()
for(i in 13:length(unique(word2vec$SubDeptName)))
{
  breakfast = word2vec[word2vec$SubDeptName == sub[i]]
  length(unique(breakfast$MicroDeptName))
  # pca
  pca <- prcomp(breakfast[,1:100, with=F], center = TRUE, scale. = TRUE)
  x = cumsum(pca$sdev)
  normalized = (x-min(x))/(max(x)-min(x))
  plot(normalized, type = "l")
  
  # choose 21
  
  nComp = 21
  pca.data = pca$x[,1:nComp]
  
  k.max = length(unique(breakfast$MicroDeptName))
  # performe cluster again
  wss <- sapply(1:k.max, 
                function(k){kmeans(pca.data, k, nstart=50,iter.max = 15 )$tot.withinss})
  
  plot(1:k.max, wss,
       type="b", pch = 19, frame = FALSE, 
       xlab="Number of clusters K",
       ylab="Total within-clusters sum of squares")
  
  cluster = kmeans(pca.data, k.max, nstart=50,iter.max = 50 )
  breakfast$cluster = i*100+cluster$cluster
  new_data = rbind(new_data, breakfast)
  print(i)

}

### merge

word2vec = word2vec%>%
  left_join(new_data[,c("SkuCode",'SubDeptName','MicroDeptName',"cluster")])

word2vec = as.data.table(word2vec)
word2vec[,micro_num :=as.numeric(as.factor(MicroDeptCode))]
fwrite(word2vec, "/Users/xiaofeifei/I/Oxford/Dissertation/train_feature.csv",row.names = FALSE)













