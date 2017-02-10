
#2017 HT Bayes Methods problem sheet 1

#####################
#Exercise 4

#load the data
a=read.table("/Users/xiaofeifei/Downloads/beetlelarva.txt")
a$easting = as.numeric(a$easting)
str(a)
attach(a)

#carry out a brief but careful prior elicitation leading to one or more priors for the parameters
#prior
K=10000; psim=rep(0,K); for (k in 1:10000) {psim[k]=rpois(1,lambda=rgamma(1,2,1))}
mean(psim>5)
# t (0,1), normal(0,1)

# Fit your models using MCMC, ABC or any other scheme that leads to a
# Bayesian quantification of uncertainty

#####################
#SAMPLE POSTERIOR p(beta|y) propto L(beta;y)p(beta)

#OK I need to write a function to evaluate the likelihood 
#for the glm L(beta;y) where beta=(beta[1],beta[2]) are the 
#GLM parameters for intercept and quarter. The likelihood 
#is a product of Poisson's so I just need the the vector of 
#means mu(beta). Let eta=beta[1]+beta[2]*quarter be the linear 
#predictor so that mu(beta)=eta(beta)^2
mu<-function(a,beta) {
  n=dim(a)[1]
  X=cbind(rep(1,n), a$northing, a$easting)
  eta=X%*%beta
  return(exp(eta))     #the sqrt link, ie mu=eta^2
  #return(exp(eta))
}

#log likelihood poisson GLM
llk<-function(a,beta) {
  return(sum(log(dpois(a$count,mu(a,beta)))))
} 

#For a prior I will use independent Cauchy (ie t(1))
#priors with scale factor 2.5, following comments in 
#Gelman et al (2010) but also thinking a bit about 
#likely and unlikely a priori increments in cases each month
#log prior t(1) scale=2.5
lpr<-function(beta) {
  sum(log(dnorm(beta,1)))
}

#MCMC setup

beta0=c(0,0,0)
#initialise (could use glm fit)

#MCMC loop - here "beta" is the current state of the Markov chain.
#betap will be the proposed state

MCMC<-function(K=10000,beta=c(0,0,0)) {
  #returns K samples from posterior using MCMC
  #no subsampling, start state goes in beta
  
  B=matrix(NA,K,3); LP=rep(NA,K)
  #storage, I will write the sampled betas here
  
  lp=llk(a,beta)+lpr(beta) 
  #log posterior is log likelihood + log prior + constant
  
  for (k in 1:K) {
    
    #tuned RW MH - I adjusted the step sizes so they were 
    #unequal for beta[1] and beta[2] 
    # !!!!!!!
    betap=beta+0.5*c(0.5,0.3,0.1)*rnorm(3) #generate candidate
    
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
K=10000
Output=MCMC(K,beta=beta0); 
B=Output$B
LP=Output$L

#check for initial transient and drop it
# pdf('ps1q4bi.pdf',9,3)
par(mfrow=c(1,2)); plot(LP[1:200],type='l'); plot(LP[seq(100,K,10)]); 
# dev.off()
#initial transient done by 100 steps - drop this burn-in
LP<-LP[100:K]
B<-B[100:K,]
par(mfrow=c(3,2))
plot(B[,1]); hist(B[,1],sqrt(K))
plot(B[,2]); hist(B[,2],sqrt(K))
plot(B[,3]); hist(B[,3],sqrt(K))


#superpose histograms from multiple runs - 
#looks OK they are consistent - here are histograms 
#of the log-lkd from multiple runs
#  change state
plot(density(LP)); 
for (i in 1:2) {
  LP.check=MCMC(K,beta=beta0)$L[100:K]
  lines(density(LP.check)); 
}

#alot of this is most easilly done using CODA
#here are some obvious functions
library(coda); 
HPDinterval(as.mcmc(B))
effectiveSize(as.mcmc(B)) 
plot(as.mcmc(B))
acf(as.mcmc(B),lag.max=200)

#plot mu(beta;quarter) against quarter
#plot cases against quarter

bayes.beta=apply(B,2,mean) #posterior mean
plot(count)
lines(y,col = "red")

mu(a,bayes.beta)


# you may be able to make a formal model comparison using Bayes factors or posterior odds, 
# but a more qualitative comparison based for example on goodness of fit may also be acceptable
# goodness of fit 
#posterior preditive checks
ecdf<-(table(c(count)))/sum(table(count))
#ecdf.mc<-(table(c(y1.mc,0:9))-1 )/sum(table(y1.mc))

ecdf.mc<-(table(c(y)))/sum(table(y))
plot(ecdf.mc,type="h",lwd=5,xlab="number of children",
     ylab=expression(paste("Pr(",italic(Y[i]==y[i]),")",sep="")),col="gray",
     ylim=c(0,.35))
points(ecdf,lwd=5,col="black",type="h")
legend('topright',
       legend=c("empirical distribution","predictive distribution"),
       lwd=c(2,2),col=
         c("black","gray"),bty="n",cex=.8)

#reasonable fit for sqrt link #posterior mean - 
#minimises the bayes risk in square error loss 

#posterior predictive distribution of RSS/var
#the variance looked a bit low so I compared the
#ppd of (y-E(y))^2/E(y) with the value on the data
n=length(a$count); y=rep(NA,n); ts=rep(NA,dim(B)[1])
for (k in 1:(dim(B)[1])) {
  mu.sim=mu(a,B[k,])
  for (j in 1:n) {
    y[j]=rpois(1,mu.sim[j])
  }
  ts[k]=mean((y-mu.sim)^2/mu.sim)
}
hist(ts,sqrt(dim(B)[1])); 
dv=mean((a$cases-mu(a,bayes.beta))^2/mu(a,bayes.beta))
abline(v=dv,col=2)
mean(ts>dv) #no issues

#post pred mean/var
mean.var.ratio.child=rep(0,1000)
for (trial in 1:1000) {
  sim1=rnbinom(n1, size=(a+s1), prob=(b+n1)/(b+n1+1))
  mean.var.ratio.child[trial]=mean(sim1)/var(sim1)
}
hist(mean.var.ratio.child,freq=F,main='',ylab='posterior predictive density')
abline(v=mean(y1)/var(y1),lwd=3)
legend('topright', c('PPD of M/V ratio','Mean/Var ratio of data'),lwd=c(1,3))