library(DNAcopy)
data(coriell)
y.all = coriell$Coriell.05296
y = y.all[900:1500]
plot(y,xlab="Genome Index", ylab="Signal")

# check NAs, total 30 NAs in y
sum(is.na(y))

# since the observation data has missing values, for dealing with missing values,
# I will partition the observation sequence into fully observed subsequences and 
# then for each subsequence either pick the most likely next state and observation.

alpha_recursion = function(y, mu, A, mean, sd, t_dis)
{
  # y is the observed sequences
  # mu is initial probability vector
  # A is the transition matrix
  # mean and sd are the emission probabilities
  # t_dis logic value, switch between t density and normal density for class1
  
  # all available states
  K = length(mu)
  
  # length of squences
  Time = length(y)
  
  # define alpha matrix to store the alphe values
  alpha = matrix(0, nrow=T,ncol=K)
  
  # chcek which density will be used for class 1
  if(t_dis == TRUE)
  {
    # calculate the alpha at time 1
    alpha[1,1] = (dt(y[1]/0.05,df = 4)/0.05)*sum(A[,1]* mu)
    for (j in 2:K) 
    {
      alpha[1,j] = dnorm(y[1],mean[j], sd[j])*sum(A[,j]* mu)
    }
    
  }else{
    
    for (j in 1:K) 
    {
      alpha[1,j] = dnorm(y[1],mean[j], sd[j])*sum(A[,j]* mu)
    }
  }
  
  # start recursion 
  for (t in 2:length) 
  {
    # check if the observed state is missing value
    # if is ture, will process forward prediction to estimated missing value
    if (is.na(y[t]))
    {
      pred_x = numeric()
      for (j in 1:K)
      {
        pred_x[j] = sum(A[,j]*(alpha[t-1,]/sum(alpha[t-1,])))
      }
      
      # find the most likely next state 
      max_x = which.max(pred_x)
      
      # simulate y 
      y[t] = rnorm(1,mean[max_x], sd[max_x])
      
      # overwrite the y
      assign("y", y, envir = .GlobalEnv) 
    }
    
    # chcek which density will be used for class 1
    if(t_dis == TRUE)
    {
      alpha[t,1] = (dt(y[t]/0.05,df = 4)/0.05)*sum(A[,1]* alpha[t-1,])
      for (j in 2:K)
      {
        alpha[t,j] = dnorm(y[t],mean[j], sd[j])*sum(A[,j]* alpha[t-1,])
      }
    }else{
      for (j in 1:K)
      {
        alpha[t,j] = dnorm(y[t],mean[j], sd[j])*sum(A[,j]* alpha[t-1,])
      }
    }
  }
  return(alpha)
}

beta_recursion = function(y, mu, A, mean, sd)
{
  # y is the observed sequences
  # mu is initial probability vector
  # A is the transition matrix
  # mean and sd are the emission probabilities
  # t_dis logic value, switch between t density and normal density for class1
  
  # all available states
  K = length(mu)
  
  # length of squences
  T = length(y)
  
  # define beta matrix to store the alphe values
  beta = matrix(0, nrow=T,ncol=K)
  
  # start recursion 
  # calculate the beta at time T
  for (j in 1:K)
  {
    beta[T,j] = 1
  }
  
  for (t in T:2)
  {
    for (i in 1:K)
    {
      # chcek which density will be used for class 1
      if(t_dis == TRUE)
      {
        B = c(dt(y[t]/0.05,df = 4)/0.05,
            dnorm(y[t],mean[2], sd[2]),
            dnorm(y[t],mean[3], sd[3]))
        
      }else{
        
        B = c(dnorm(y[t],mean[1], sd[1]),
            dnorm(y[t],mean[2], sd[2]),
            dnorm(y[t],mean[3], sd[3]))
      }
      beta[t-1,i] = sum(B*A[i,]* beta[t,])
    }
  }
  return(beta)
}

# define transition matrix, emission probabilities, probability vector
A = matrix(c(0.99, 0.005, 0.005, 0.005, 0.99, 0.005, 0.005, 0.005, 0.99),
           nrow = 3, ncol = 3, byrow = T)
mean = c(0,0.5,-0.5)
sd = c(0.05, 0.08,0.08)
mu = c(1/3,1/3,1/3)

# run alpha_recursion and beta_recursion
#normal density
a_n = alpha_recursion(y, mu, A, mean, sd, FALSE)
b_n = beta_recursion(y, mu, A, mean, sd,FALSE)

# t density
a_t = alpha_recursion(y, mu, A, mean, sd, TRUE)
b_t = beta_recursion(y, mu, A, mean, sd, TRUE)
# smoothing, find marginal maximum a posteriori
smoothing_n = a_n*b_n/apply(a_n*b_n, 1, sum)
estimate_state_n = apply(smoothing_n, 1, which.max)

smoothing_t = a_t*b_t/apply(a_t*b_t, 1, sum)
estimate_state_t = apply(smoothing_t, 1, which.max)

# Define expectation-maximisation algorithm and estimate the transition matrix A
EM = function(alpha, beta, A, mu, mean, sd)
{
  # alpha, beta are the result of the forward and backward recursions
  # A is transition matrix
  # mu is the initial probability vector
  # mean and sd are the emission probabilities
  
  # all available states
  K = length(mu)
  # length of squences
  Time = dim(alpha)[1]
  
  # define expected matrix to store the E[Ni,j |y1:T , A(k−1)]
  E = matrix(0,nrow = 3, ncol = 3)
  for(i in 1:K)
  {
    for(j in 1:K)
    {
      for(t in 2:Time)
      {
        # E[Ni,j |y1:T , A(k−1)]
        E[i,j] = E[i,j] + alpha[t-1,i]*dnorm(y[t],mean[j], sd[j])*A[i,j]*beta[t,j]
      }
    }
  }
  
  # Ai,j = E[Ni,j |y1:T , A(k−1)]/sum_l(E[Ni,l |y1:T , A(k−1)])
  A = E/apply(E, 1,sum)
  return(A)
}

# define transition matrix, emission probabilities, probability vector, likelihood
A = matrix(rep(0.5,9), nrow = 3, ncol = 3, byrow = T)
mean = c(0,0.5,-0.5)
sd = c(0.05, 0.08,0.08)
mu = c(1/3,1/3,1/3)
likelihood = 0

while(TRUE)
{
  # calculate forward and backward recursions
  a = alpha_recursion(y, mu, A, mean, sd, FALSE)
  b = beta_recursion(y, mu, A, mean, sd, FALSE)
  
  # calculate likelihood with new transition matrix
  likelihoodnew = sum(a[dim(a)[1],])
  if (likelihoodnew<likelihood)
  {
    break
    
  }else{
    likelihood = likelihoodnew
  }
  
  # apply EM algorithm
  A = EM(a, b, A, mu, mean, sd)
}

###############################################################################
###############################################################################

# plot state and y
plot(y,xlab="Genome Index", ylab="Signal", col = estimate_state)
ind = which(state==1)
points(x = ind,y = rep(0,length(ind)), col='red', pch=15)

ind = which(state==2)
points(x = ind,y = rep(0.5,length(ind)), col='green', pch=15)

ind = which(state==3)
points(x = ind,y = rep(-0.6,length(ind)), col='blue', pch=15)

plot(y,xlab="Genome Index", ylab="Signal", col = estimate_state1)
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

