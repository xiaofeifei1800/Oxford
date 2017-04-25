library(DNAcopy)
data(coriell)
y.all = coriell$Coriell.05296
y = y.all[900:1500]
plot(y)

# check NAs, total 30 NAs in y
sum(is.na(y))

# Exercise 1 (a)


# since the observation data has missing values, for dealing with missing values,
# I will partition the observation sequence into fully observed subsequences and 
# then for each subsequence either pick the most likely next state and observation.

alpha_recursion = function(y, mu, A, mean, sd)
{
  K = length(mu)
  T = length(y)
  alpha = matrix(0, nrow=T,ncol=K)
  
  for (j in 1:K) 
  {
      if (y[1]>mean[j])
      {
        alpha[1,j] = pnorm(y[1],mean[j], sd[j], lower.tail = F)*sum(A[,j]* mu)
      }else{
        alpha[1,j] = pnorm(y[1],mean[j], sd[j])*sum(A[,j]* mu)
      }
  }
  
  for (t in 2:T) 
  {
    if (is.na(y[t]))
    {
      pred_x = numeric()
      for (j in 1:K)
      {
        pred_x[j] = sum(A[,j]*(alpha[t-1,]/sum(alpha[t-1,])))
        # pred_y = rnorm(1,mean[j], sd[j])
      }
      max_x = which.max(pred_x)
      y[t] = rnorm(1,mean[max_x], sd[max_x])
    }
    for (j in 1:K)
    {
      if (y[t]>mean[j])
      {
        alpha[t,j] = pnorm(y[t],mean[j], sd[j], lower.tail = F)*sum(A[,j]* alpha[t-1,])
      }else{
        alpha[t,j] = pnorm(y[t],mean[j], sd[j])*sum(A[,j]* alpha[t-1,])
      }
    }
  }
  return(alpha)
}


A = matrix(c(0.99, 0.005, 0.005, 0.005, 0.99, 0.005, 0.005, 0.005, 0.99),
           nrow = 3, ncol = 3, byrow = T)

mean = c(0,0.5,-0.5)
sd = c(0.05, 0.08,0.08)
mu = c(1/3,1/3,1/3)
a = alpha_recursion(y, mu, A, mean, sd)







beta_recursion = function(y, mu, mean, sd)
{
  K = length(mu)
  T = length(y)
  beta = matrix(0, nrow=T,ncol=K)
  for (j in 1:K)
  {
    beta[T,j] = 1
  }
  
  for (t in T:2)
  {
    for (i in 1:K)
    {
      beta[t-1,i] = sum(B[,y[t]]*A[i,]* beta[t,])
    }
  }
  return(beta)
}






#
viterbi = function(y, mu, A, B)
{
  K = length(mu)
  T = length(y)
  m = matrix(0, nrow=T,ncol=K)
  x.map = rep(0, T)
  # Backward
  for (i in 1:K) m[T,i] = 1
  for (t in T:2) for (i in 1:K) m[t-1,i] = max(B[,y[t]]*A[i,]* m[t,])
  #Forward
  x.map[1] = which.max(m[1,] * B[,y[1]] * mu)
  for (t in 2:T) x.map[t] = which.max(m[t,]*B[,y[t]]*A[x.map[t-1],])
  return(x.map)
}
