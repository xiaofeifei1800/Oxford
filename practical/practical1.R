#3.1

caffe = read.csv("P:/R Data/caffeine.csv")
summary(caffe)
apply(caffe, 2, sd)

#3.2
# change structure of data
temp = c(caffe$None,caffe$Caff.100ml, caffe$Caff.200ml)
cata = factor(rep(c("None", "Caff.100ml","Caff.200ml"), each = 10))
new_caffe = as.data.frame(cbind(temp, cata))

boxplot(caffe)

ggplot(new_caffe, aes(x=temp, colour=cata)) +
  geom_histogram(binwidth = 5)


