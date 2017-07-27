library(data.table)
pred = fread('/Users/xiaofeifei/I/Oxford/Dissertation/some.csv')

combine = aggregate(pred[, 1:20], list(pred$V21), mean)

combine = t(combine)
combine = combine[-1,]

# boxplot
boxplot(combine)

# CI
CI = matrix(nrow = 10, ncol = 2)
for(i in 1:10)
{
  a = t.test(combine[,i], conf.level = 0.95)
  conf = a$conf.int[1:2]
  CI[i,]=conf
}


