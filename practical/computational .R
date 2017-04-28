library(DNAcopy)
data(coriell)
y.all = coriell$Coriell.05296
y = y.all[900:1500]
plot(y,xlab="Genome Index", ylab="Signal")

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
        alpha[1,j] = dnorm(y[1],mean[j], sd[j])*sum(A[,j]* mu)

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
      assign("y", y, envir = .GlobalEnv) 
    }
    for (j in 1:K)
    {
        alpha[t,j] = dnorm(y[t],mean[j], sd[j])*sum(A[,j]* alpha[t-1,])
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


beta_recursion = function(y, mu, A, mean, sd)
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
      B = c(dnorm(y[t],mean[1], sd[1]),
            dnorm(y[t],mean[2], sd[2]),
            dnorm(y[t],mean[3], sd[3]))
      beta[t-1,i] = sum(B*A[i,]* beta[t,])
    }
  }
  return(beta)
}
b = beta_recursion(y, mu, A, mean, sd)

# smoothing, marginal maximum a posteriori
smooth = a*b/apply(a*b, 1, sum)
state = apply(smooth, 1, which.max)

# Exercise 2
# 2.1


alpha_recursion = function(y, mu, A, mean, sd)
{
  K = length(mu)
  T = length(y)
  alpha = matrix(0, nrow=T,ncol=K)
  alpha[1,1] = (dt(y[1]/0.05,df = 4)/0.05)*sum(A[,1]* mu)
  for (j in 2:K) 
  {
    alpha[1,j] = dnorm(y[1],mean[j], sd[j])*sum(A[,j]* mu)
    
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
      assign("y", y, envir = .GlobalEnv) 
    }
    alpha[t,1] = (dt(y[t]/0.05,df = 4)/0.05)*sum(A[,1]* alpha[t-1,])
    for (j in 2:K)
    {
      alpha[t,j] = dnorm(y[t],mean[j], sd[j])*sum(A[,j]* alpha[t-1,])
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


beta_recursion = function(y, mu, A, mean, sd)
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
      B = c(dt(y[t]/0.05,df = 4)/0.05,
            dnorm(y[t],mean[2], sd[2]),
            dnorm(y[t],mean[3], sd[3]))
      beta[t-1,i] = sum(B*A[i,]* beta[t,])
    }
  }
  return(beta)
}
b = beta_recursion(y, mu, A, mean, sd)

# smoothing, marginal maximum a posteriori
smooth = a*b/apply(a*b, 1, sum)
state1 = apply(smooth, 1, which.max)

# plot state and y
returns.hv = as.data.frame(y)
returns.hv$HighVolatility = state

plot(y,xlab="Genome Index", ylab="Signal", col = state)
ind = which(state==1)
points(x = ind,y = rep(0,length(ind)), col='red', pch=15)

ind = which(state==2)
points(x = ind,y = rep(0.5,length(ind)), col='green', pch=15)

ind = which(state==3)
points(x = ind,y = rep(-0.6,length(ind)), col='blue', pch=15)

plot(y,xlab="Genome Index", ylab="Signal", col = state)
ind = which(state1==1)
points(x = ind,y = rep(0,length(ind)), col='red', pch=15)

ind = which(state1==2)
points(x = ind,y = rep(0.5,length(ind)), col='green', pch=15)

ind = which(state1==3)
points(x = ind,y = rep(-0.6,length(ind)), col='blue', pch=15)

# qq
qqnorm(y[state1==1], main="Low volatility")
qqline(y[state1==1])

qqnorm(y[state1==2], main="Low volatility")
qqline(y[state1==2])

qqnorm(y[state1==3], main="Low volatility")
qqline(y[state1==3])

# EM
A_inital = A = matrix(rep(0.5,9), nrow = 3, ncol = 3, byrow = T)
mu = c(1/3,1/3,1/3)
mean = c(0,0.5,-0.5)
sd = c(0.05, 0.08,0.08)
likelihood = 0
for(i in 1:10)
{
  a = alpha_recursion(y, mu, A, mean, sd)
  b = beta_recursion(y, mu, A, mean, sd)
  
  K = length(mu)
  Time = length(y)
  E = matrix(0,nrow = 3, ncol = 3)
  for(i in 1:K)
  {
    for(j in 1:K)
    {
      for(t in 2:Time)
      {
        E[i,j] = E[i,j] + a[t-1,i]*dnorm(y[t],mean[j], sd[j])*A[i,j]*b[t,j]
      }
    }
  }
  
  A = E/apply(E, 1,sum)
  likelihoodnew = sum(a[dim(a)[1],])
  if (likelihoodnew<likelihood)
  {
    break
  }
}

#############################################################################

A_inital = A = matrix(rep(0.5,9), nrow = 3, ncol = 3, byrow = T)
mu = c(1/3,1/3,1/3)
mean = c(0,0.5,-0.5)
sd = c(0.05, 0.08,0.08)
likelihood = 0
for(i in 1:10)
{
  a = alpha_recursion(y, mu, A, mean, sd)
  b = beta_recursion(y, mu, A, mean, sd)
  
  K = length(mu)
  Time = length(y)
  E = matrix(0,nrow = 3, ncol = 3)
  for(i in 1:K)
  {
    for(j in 1:K)
    {
      for(t in 2:Time)
      {
        if (j == 1)
        {
          E[i,j] = E[i,j] + a[t-1,i]*(dt(y[t]/0.05,df = 4)/0.05)*A[i,j]*b[t,j]
        }else{
          E[i,j] = E[i,j] + a[t-1,i]*dnorm(y[t],mean[j], sd[j])*A[i,j]*b[t,j]
        }
      }
    }
  }
  
  A = E/apply(E, 1,sum)
  likelihoodnew = sum(a[dim(a)[1],])
  if (likelihoodnew<likelihood)
  {
    break
  }
}


