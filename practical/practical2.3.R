#load the data
a=read.table("/Users/xiaofeifei/Downloads/beetlelarva.txt")

# change easting from factor to numerical data
a$easting = as.numeric(a$easting)
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

# Function to evaluate the likelihood for the glm L(beta;y) where beta=(beta[1],beta[2],beta[3]) 
# are the GLM parameters for intercept, northing and easting. The likelihood is a product of Poisson's 
# Let eta=beta[1]+beta[2]*northing+beta[2]*easting be the linear predictor so that mu(beta)=exp(eta(beta))

mu<-function(a,beta) {
  n=dim(a)[1]
  X=cbind(rep(1,n), a$northing, a$easting)
  eta=X%*%beta
  return(exp(eta))    
}

#log likelihood poisson GLM
llk<-function(a,beta) {
  return(sum(log(dpois(a$count,mu(a,beta)))))
} 


# use prior normal mean = 0, sd = 1
lpr<-function(beta) {
  sum(log(dnorm(beta)))
}

#MCMC setup

#initialise 
beta0=c(0,0,0)


#MCMC loop - here "beta" is the current state of the Markov chain.
#betap will be the proposed state

MCMC<-function(K=10000,beta=c(0,0,0)) {
  #returns K samples from posterior using MCMC
  #no subsampling, start state goes in beta
  
  B=matrix(NA,K,3); LP=rep(NA,K)
  #storage
  
  lp=llk(a,beta)+lpr(beta) 
  #log posterior is log likelihood + log prior + constant
  
  for (k in 1:K) {
    
    #tuned RW MH - adjusted the step sizes so they were 
    #unequal for beta[1] and beta[2] 
    ##########################################
    ## why use different step size for beta ##
    ##########################################
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
#################################
## how to select drop out point##
#################################

# pdf('ps1q4bi.pdf',9,3)
par(mfrow=c(1,2)); plot(LP[1:200],type='l'); plot(LP[seq(100,K,10)]); 
# dev.off()
#initial transient done by 100 steps - drop this burn-in
LP<-LP[100:K]
B<-B[100:K,]

#superpose histograms from multiple runs - 
#looks OK they are consistent - here are histograms 
#of the log-lkd from multiple runs change state
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

# you may be able to make a formal model comparison using Bayes factors or posterior odds, 
# but a more qualitative comparison based for example on goodness of fit may also be acceptable

# goodness of fit 
# posterior preditive checks
# plot ECDF of data and predictive distribution
ecdf<-(table(c(count)))/sum(table(count))

ecdf.mc<-(table(c(y)))/sum(table(y))

plot(ecdf.mc,type="h",lwd=5,xlab="number of bettle",
     ylab=expression(paste("Pr(",italic(Y[i]==y[i]),")",sep="")),col="gray",
     ylim=c(0,.35))
points(ecdf,lwd=5,col="black",type="h")
legend('topright',
       legend=c("empirical distribution","predictive distribution"),
       lwd=c(2,2),col=
         c("black","gray"),bty="n",cex=.8)

#posterior predictive distribution of RSS/var
#the variance looked a bit low so I compared the
#ppd of (y-E(y))^2/E(y) with the value on the data
########################################
## RSS/var, mean/var, (y-E(y))^2/E(y) ##
########################################
n=length(a$count); y=rep(NA,n); ts=rep(NA,dim(B)[1])
for (k in 1:(dim(B)[1])) {
  mu.sim=mu(a,B[k,])
  for (j in 1:n) {
    y[j]=rpois(1,mu.sim[j])
  }
  ts[k]=mean(y)/var(y)
}
hist(ts,sqrt(dim(B)[1])); 
bayes.beta = apply(B,2,mean)
dv=mean(count)/var(count)
abline(v=dv,col=2)
mean(ts>dv) #no issues

