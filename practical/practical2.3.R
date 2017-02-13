library(ggplot2)
library(xtable)
#load the data
a=read.table("P:/R Data/beetlelarva.txt")

# change easting from factor to numerical data
a$easting = factor(a$easting)
a$northing = factor(a$northing)

attach(a)

ggplot(data = a, aes(x = easting, y = northing)) +
  geom_tile(aes(fill = count))+
  xlab("Easting")+ ylab("Northing")+
  ggtitle("Heat map of Japanese beetle larvae")

sum_northing = aggregate(a$count, by=list(Category=a$northing), FUN=sum)
sum_easting = aggregate(a$count, by=list(Category=a$easting), FUN=sum)

#carry out a brief but careful prior elicitation leading to one or more priors for the parameters
#Using simulation in prior elicitation
K=1000
n=length(a$count); mu.sima=matrix(NA,K,n)
for (k in 1:K) {
  lambda=rnorm(1,2,1)
  gamma=rnorm(12,0,1)
  beta = c(lambda,gamma)
  mu.sim = mu(X,beta)
  for (j in 1:n) {
    mu.sima[k,j]=mu.sim[j]
  }
}
par(mfrow = c(1,2))
plot(apply(mu.sima,2,mean), xlab = "cell number", ylab = "simulated values")


# Fit your models using MCMC, ABC or any other scheme that leads to a
# Bayesian quantification of uncertainty

#####################
#SAMPLE POSTERIOR p(beta|y) propto L(beta;y)p(beta)

# Function to evaluate the likelihood for the glm L(beta;y) where beta=(beta[1],beta[2],beta[3])
# are the GLM parameters for intercept, northing and easting. The likelihood is a product of Poisson's
# Let eta=beta[1]+beta[2]*northing+beta[2]*easting be the linear predictor so that mu(beta)=exp(eta(beta))
X=model.matrix(~northing+easting,data = a)
mu<-function(X,beta) {
  n=dim(a)[1]
  eta=X%*%beta
  return(exp(eta))
}

#log likelihood poisson GLM
llk<-function(a,beta) {
  return(sum(dpois(a$count,mu(X,beta), log = T)))
}


# use prior normal mean = 0, sd = 1
lpr<-function(beta) {
  sum(dnorm(beta[2:13],0,log = T), dnorm(beta[1],1,1,log = T))
}

#MCMC setup

#initialise
beta0=rep(0,13)


#MCMC loop - here "beta" is the current state of the Markov chain.
#betap will be the proposed state

MCMC<-function(K=10000,beta=c(0,0,0)) {
  #returns K samples from posterior using MCMC
  #no subsampling, start state goes in beta

  B=matrix(NA,K,13); LP=rep(NA,K)
  #storage

  lp=llk(a,beta)+lpr(beta)
  #log posterior is log likelihood + log prior + constant

  for (k in 1:K) {

    #tuned RW MH - adjusted the step sizes so they were
    #unequal for beta[1] and beta[2]
    ##########################################
    ## why use different step size for beta ##
    ##########################################
    betap=beta+0.5*c(0.5,rep(0.1,12))*rnorm(13) #generate candidate

    lpp=llk(a,betap)+lpr(betap)        #calculate log post for candidate

    MHR=lpp-lp                         #"log(pi(y)/pi(x))" from lectures

    if (log(runif(1))<MHR) {           #Metropolis Hastings acceptance step
      beta=betap                       #if we accept update the state
      lp=lpp
    }

    B[k,]=beta                         #save the sequence of MCMC states, our samples.
    LP[k]=lp
  }
  return(list(B=B,L=LP))
}

#basic plotting
K=200000
Output=MCMC(K,beta=beta0);
B=Output$B
LP=Output$L

#check for initial transient and drop it
par(mfrow=c(1,2))
plot(LP[1:400],type='l', xlab = "iteration", ylab = "log posterior")
plot(LP[seq(400,K,10)],xlab = "iteration", ylab = "log posterior");

#initial transient done by 100 steps - drop this burn-in
LP<-LP[100:K]
B<-B[100:K,]

#superpose histograms from multiple runs -
#looks OK they are consistent - here are histograms
#of the log-lkd from multiple runs change state
beta1 = matrix(c(rep(1,13), rep(-1,13)),2,13, byrow = T)
par(mfrow = c(1,1))
plot(density(LP), xlab = "log posterior", main = "");
for (i in 1:2) {
  LP.check=MCMC(K,beta=beta1[i,])$L[100:K]
  lines(density(LP.check), col=i+1);
}

#alot of this is most easilly done using CODA
#here are some obvious functions

# 关于链的收敛有这样一些检验方法。
# （1）图形方法 这是简单直观的方法。我们可以利用这样一些图形：
#  （a）迹图（trace plot）：将所产生的样本对迭代次数作图，生成马氏链的一条样本路径。如果当t足够大时，
# 路径表现出稳定性没有明显的周期和趋势，就可以认为是收敛了。
# (b)自相关图（Autocorrelation plot）：如果产生的样本序列自相关程度很高，用迹图检验的效果会比较差。
# 一般自相关随迭代步长的增加而减小，如果没有表现出这种现象，说明链的收敛性有问题。
library(coda);
temp = HPDinterval(as.mcmc(B))
effectiveSize(as.mcmc(B))

par(mfrow=c(1,2))
for (i in 1:2)
{
plot(B[,i], type = "l", ylab = "sample value", xlab = "iteration",
     main = paste0(c("Beta "), i, c( "trace plot")))
}

par(mfrow=c(4,4))
for (i in 1:13)
{
  acf(as.mcmc(B[,i]),lag.max=3000)
}


# you may be able to make a formal model comparison using Bayes factors or posterior odds,
# but a more qualitative comparison based for example on goodness of fit may also be acceptable

# goodness of fit
# posterior preditive checks
# plot ECDF of data and predictive distribution
par(mfrow = c(1,1))
n=length(a$count); y=matrix(NA,dim(B)[1],n)
for (k in 1:(dim(B)[1])) {
  mu.sim=mu(X,B[k,])
  for (j in 1:n) {
    y[k,j]=rpois(1,mu.sim[j])
  }
}

bayes.beta = apply(B,2,mean)
mu.sim=mu(X,bayes.beta)
n=length(a$count); y=rep(NA,n)
for (j in 1:n) {
  y[j]=rpois(1,mu.sim[j])
}
ecdf<-(table(c(count)))/sum(table(count))

ecdf.mc<-(table(c(y)))/sum(table(y))
scatterplot3d(b, type = "h", y.margin.add = 0.2, box = F)
plot(ecdf,type="h",lwd=5,xlab="number of beetle",
     ylab=expression(paste("Pr(",italic(Y[i]==y[i]),")",sep="")),col="gray",
     ylim=c(0,.35))
points(a+0.3,ecdf.mc,lwd=5,col="black",type="h")
legend('topright',
       legend=c("empirical distribution","predictive distribution"),
       lwd=c(2,2),col=
         c("black","gray"),bty="n",cex=.8)

#posterior predictive distribution of RSS/var
#We are working with a Poisson, so I chose to check that predictive
#distribution of the mean/variance ratio and compare it with
#the value we got
########################################
## RSS/var, mean/var, (y-E(y))^2/E(y) ##
########################################
n=length(a$count); y=rep(NA,n); ts=rep(NA,dim(B)[1])
for (k in 1:(dim(B)[1])) {
  mu.sim=mu(X,B[k,])
  for (j in 1:n) {
    y[j]=rpois(1,mu.sim[j])
  }
  ts[k]=mean(y)/var(y)
}
hist(ts,sqrt(dim(B)[1]), xlab = "ratio between mean and variance", main = "");

dv=mean(count)/var(count)
abline(v=dv,col=2)
legend('topright', c('PPD of M/V ratio', 'Mean/Var ratio of data'), lwd=c(1,3))
mean(ts>dv) #no issues

###############################
b.p=MCMCprobit(fail~temp,b0=c(0,0),B0=c(3,3),seed=4,mcmc=N,marginal.likelihood='Laplace')
b.l=MCMClogit(fail~temp,b0=c(0,0),B0=c(3,3),seed=5,mcmc=N,marginal.likelihood='Laplace')

logit<-function(beta,x=temp) {eta=beta[1]+beta[2]*x; exp(eta)/(1+exp(eta))}
probit<-function(beta,x=temp) {eta=beta[1]+beta[2]*x; pnorm(eta)}

lkd<-function(beta,logit=T) {
  if (logit) {pb=logit(beta)} else {pb=probit(beta)}
  return(sum(dbinom(fail,1,pb,log=T)))
}

l.pr=l.lo=pr.lo=pr.pr=l.lo.pr=l.pr.lo=rep(NA,N)
for (k in 1:N) {
  like.log[k]=lkd(b.l[k,],logit=T);
  like.pro[k]=lkd(b.p[k,],logit=F);
  like.log.pro[k]=lkd(b.p[k,],logit=T);
  like.pro.log[k]=lkd(b.l[k,],logit=F);
  prior.log[k]=sum(dnorm(b.l[k,],0,9,log=T))
  prior.pro[k]=sum(dnorm(b.p[k,],0,9,log=T))
}

#harmonic
ml.lo=1/mean(1/exp(l.lo))
ml.pr=1/mean(1/exp(l.pr))
ml.lo/ml.pr

#Bridge estimator just using h=1 - simple, performs well
mean(exp(l.lo.pr)*pr.pr)/mean(exp(l.pr.lo)*pr.lo)

#builtin function uses a Laplace estimator we didnt cover
exp(attr(b.l,"logmarglike"))/exp(attr(b.p,"logmarglike"))

#the naive estimator is hopeless
N=1000000
b.prior=matrix(rnorm(2*N,0,9),N,2)
l.pr=l.lo=rep(NA,N)
for (k in 1:N) {l.lo[k]=lkd(b.prior[k,],logit=T); l.pr[k]=lkd(b.prior[k,],logit=F)}
mean(exp(l.lo))/mean(exp(l.pr))



