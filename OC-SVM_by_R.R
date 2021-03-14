# final project

#problem 1#######################

#problem 1a: compute the dual problem and obtain alpha vector
#######################
require(quadprog)
library(quadprog)
library(psych)
data=read.delim(file.choose(),header=FALSE, sep="")
dim(data)
#partition data into training and test parts
train=data[data[1]==1,]
trainx=train[,-1]
trainy=train[,1]
test=data[data[1]==2,]
testx=test[,-1]
testy=test[,1]
n=dim(trainx)[1] # the total number of training set
# initialize all matrices and vectors required by QP solver
# define the kernel function
C=0.1
sigma=1

kern=function(xi,xj,sigma){ 
  #three input
  exp(-(dist(rbind(xi,xj))[1])^2/sigma^2)
}
#formulate H matrix
H_matrix=matrix(0,n,n)
for (i in 1:n){
  for (j in 1:n){
    H_matrix[i,j]=kern(trainx[i,],trainx[j,],sigma)
  }
}
#formulate d vector
dvec=as.vector(array(0,n))
#formulate A matrix
Amat=t((rbind(t(trainy),diag(n),-diag(n))))
#formulate b0 vector
b0=rbind(1,matrix(0,n),C*matrix(-1,n))
alpha=solve.QP(H_matrix,dvec,Amat,bvec=b0)$solution
sum(alpha)
alpha

#problem 1b:test the model performance
#######################

#compute the average support vector p
p=sum(H_matrix %*% as.matrix(alpha))/n  # use vectorization instead of loop

#define the prediction function
pred=function(xnew,trainx,alpha,p){
  kvec=rep(0,dim(trainx)[1])
  for(i in 1:dim(trainx)[1]){
    kvec[i]=kern(xnew,trainx[i,],sigma)
  }
  f=sign((t(alpha)%*%kvec)[1]-p)
  return(f)
}
#test the predictive accuracy
t2=0 # for test accuracy
for (i in 1:dim(testx)[1]){
  if(pred(testx[i,],trainx,alpha,p)!=1){
    t2=t2+1
  }
}
cat("test accuracy:",t2/dim(testx)[1])


#problem 2#######################
#######################

#probelm 2a:a function to update inverse matrix Hn-1
#######################

#read the dataset
charlie=read.csv(file.choose(),header=T)
data=charlie[,c("Data","x1","x2","x3","x4")]
#partition data into training and test sets
train=data[data["Data"]=='Original',]
trainx=train[,-1]
trainy=train[,1]
test=data[data["Data"]=='New',]
testx=test[,-1]
testy=test[,1]
# define the kernel function
k=function(xi,xj,sigma){ 
  #three input
  exp(-(dist(rbind(xi,xj))[1])^2/sigma^2)
}
#define a function to update the inverse of Hn matrix
Hn_inv=function(hn_1_inv,data,xn){ # three inputs: the inverse of Hn-1,(n-1)iteration data,new sample Xn
  d=dim(data)[1]      # the number of observation
  delta_n=matrix(0,d) # column vector delta_n={k(xn,xi)}
  for (i in 1:d){
  delta_n[i]=k(xn,data[i,],sigma)
  }
  delta_nn=k(xn,xn,sigma)   # the scalar delta_nn by xn itself
  an=hn_1_inv %*% as.matrix(delta_n)   # (n-1)*1 vector an
  rn=delta_nn+(1/(2*C))-(t(delta_n) %*% an)[1]
  hn_inv=(1/rn)*rbind(cbind(rn*hn_1_inv+an %*% t(an),-an),cbind(-t(an),1))
return(hn_inv)
}
#test Hn_inv function by the inverse of H1 and H2  
library(matlib)
C=0.01  # the penalty factor
sigma=10 # the parameter of Gaussian Kernel

h1=k(trainx[1,],trainx[1,],sigma)+1/(2*C)
h1_inv=as.matrix(1/h1) # the inverse of H1
h2_inv=Hn_inv(h1_inv,trainx[1,],trainx[2,]) # the inverse of H2
#a loop for computing the inverse of  Hn
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
n=4                # assum the number of observation=4
for (i in 1:n){
cat("n=",i,",","inverse","of","H",i,"=","\n")
print(hn_1_inv)
hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[i+1,])
hn_1_inv=hn_inv
}

#problem 2b:Compare the training times (in sec) between the recursive and direct
#######################

# iterative method
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=19  # assume we have 19 observations
t1=rep(0,N)              # record the timing as n increases for iterative method
start_time1 = Sys.time()
for (i in 1:N){
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[i+1,])
  hn_1_inv=hn_inv
  t1[i]=Sys.time()-start_time1  
  cat("iterative method timing:","H",i+1,",",Sys.time()-start_time1, "seconds","\n")
}

# direct approach
#define a function of computing Hn directly
Hn=function(data){
  d=dim(data)[1] # the number of observation
  Km=matrix(0,d,d) # K matrix named as Km
  I=diag(d) # d dimensions identity matrix
for (i in 1:d){
  for (j in 1:d) {
  Km[i,j]=k(data[i,],data[j,],sigma)}
}
hn=Km+(1/(2*C))*I # compute the matrix Hn
return (hn)
}

N=19  # assume we have 19 observations
t2=rep(0,N)    # record the timing as n increases for direct method
start_time2 = Sys.time()
for (i in 1:N){
  hn_inv=inv(Hn(trainx[1:(i+1),]))
  t2[i]=Sys.time()-start_time2
  cat("direct approach timing:","H",i+1,",",Sys.time()-start_time2,"seconds","\n")
}
#compare the runtimes of two method
x <- rep(1:N)
plot(x,t2,type="l",col="blue",xlab="the number of iterations",
      ylab="Runtime (s)", main="Runtime of the direct and iterative methods")
#Add more data to the plot
lines(x,t1,col="red")  #add runtimes of iterative method
legend(x=1,y=7,c("direct method","iterative method"),cex=1.1, 
         col=c("blue","red"),lty=c(1,1)) #add legend

#problem 2c:update the solution alpha vector
#######################

#define a function to compute alpha1 vector
alpha1=function(hn_inv,data){ 
  # two inputs: the inverse of Hn and n observations
  d=dim(data)[1]   # the number of observation
  e=rep(1,d) # n*1 column vector
  kj=rep(0,d) # n*1, kn vector
for(i in 1:d){
  kj[i]=k(data[i,],data[i,],sigma)
}
p1=2-t(e) %*% hn_inv %*% kj
p2=t(e) %*% hn_inv %*% e
alpha=0.5*hn_inv %*% (kj + (p1[1]/p2[1])*e)
return (alpha)
}

hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=4  # assume we have 4 observations
for (i in 1:N){
  alpha_n=alpha1(hn_1_inv,trainx[1:i,])
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[i+1,])
  hn_1_inv=hn_inv
cat("n=",i,",","alpha=","\n",alpha_n,"\n")
}

#problem 2d:compare alpha_n in iterative fashion and in direct approach
#######################

# iterative method
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=7  # assume we have 7 observations
for (i in 1:N) {
  alpha_n=alpha1(hn_1_inv,trainx[1:i,])
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[i+1,])
  hn_1_inv=hn_inv
  cat("iterative method:n=",i,",","alpha=","\n",alpha_n,"\n")
}
# direct approach
hn_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=7  # assume we have 7 observations
for (i in 1:N){
  alpha_n=alpha1(hn_inv,trainx[1:i,])
  hn_inv=inv(Hn(trainx[1:(i+1),]))
  cat("direct method:n=",i,",","alpha=","\n",alpha_n,"\n") 
}


#problem 2e:update the radius R2n and dz
#######################

## assume we have a train set with 7 observations and a test set with 7 observations
# compute R2 from the training dataset    
R2=function(alpha_n,train){
  # R2 consists of three average parts:ap1,ap2,ap3 
  d=dim(train)[1]
  ap1=0
  ap2=0
  ap3=0
for (i in 1:d){
  ap1=ap1+k(train[i,],train[i,],sigma)  }

for (i in 1:d){
  for (j in 1:d){
  ap2=ap2+2*alpha_n[j]*k(train[i,],train[j,],sigma) }
}
for( i in 1:d){
  for (j in 1:d){
  ap3=ap3+alpha_n[i]*alpha_n[j]*k(train[i,],train[j,],sigma) }
}
r2=(ap1-ap2)/d+ap3
return(r2)
}
#compute dz of a test dataset
DZ=function(alpha_n,train,test){
  # assume dz consists of three parts:p1,p2,p3
  d=dim(train)[1]
  p1=k(test,test,sigma)
  p2=0
  p3=0
for (i in 1:d){
  p2=p2+2*alpha_n[i]*k(test,train[i,],sigma)}
for (i in 1:d){
  for(j in 1:d){
  p3=p3+alpha_n[i]*alpha_n[j]*k(train[i,],train[j,],sigma) }
}
dz=p1-p2+p3
return(dz)
}
C=0.01
sigma=10
tn1=7
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
for (i in 1:tn1){ 
  alpha_n=alpha1(hn_1_inv,trainx[1:i,])
  r2=R2(alpha_n,trainx[1:i,])
  dz=DZ(alpha_n,trainx[1:i,],trainx[(i+1),])
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[(i+1),])
  hn_1_inv=hn_inv
  cat("n=",i,"\n","R2=",r2,"\n","dz=",dz,"\n")
}
#check the next 7 rows
tn2=14
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
for (i in 1:tn2){ 
  alpha_n=alpha1(hn_1_inv,trainx[1:i,]) # iterative alpha_n
  r2=R2(alpha_n,trainx[1:i,])           # iterative R2 (n)
  dz=DZ(alpha_n,trainx[1:i,],trainx[1:(i+1),]) # dz of test vector xn
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[(i+1),]) #iterative inv(Hn)
  hn_1_inv=hn_inv
  if (i>6){   #predict the next 7 rows
    cat("n=",i,"\n","R2=",r2,"\n","dz=",dz,"\n")
  if (dz<=r2){
    print("It's a target")}
  else {
    print("It's an outlier")}
  }
}

#problem 3#######################
#######################

#problem 3a: obtain the inverse of H10
#######################

# iterative method
h1=k(trainx[1,],trainx[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
hn_1_inv=h1_inv    # the inverse of Hn-1 starts from the inverse of H1
N=9  # assume we add 9 observations sequentially
for (i in 1:N){
  hn_inv=Hn_inv(hn_1_inv,trainx[1:i,],trainx[i+1,])
  hn_1_inv=hn_inv
}
cat("iterative method:","inverse of H10=","\n")
print(hn_inv)

#problem 3b: add a new sample and remove an oldest sample
#######################

#define a function to update the inverse of U10 in eq(9)
U_inv=function(h10_inv,traindata,newdata){
  d=dim(newdata)[1]
  for (i in 1:d){
  h11_inv=Hn_inv(h10_inv,traindata,newdata[i,])
  row1=h11_inv[1,]   # first row of h11_inv
  row10=h11_inv[-1,] # last ten rows of h11_inv
  e=row1[1]   # the constant e in block of Hn
  ft=row1[-1] # the transpose of f vector
  f=row10[,1] # f vector
  D=row10[,-1] # D matrix
  U10_inv=D-(1/e)*f%*%t(ft) # inverse of U10 in eq(9)
  # remove the oldest one and add a new one
  traindata=rbind(traindata[-1,],newdata[i,]) 
  h10_inv=U10_inv
  }
  return(U10_inv)
}
h10_inv=hn_inv # the inverse of H10 at first time
U10_inv=U_inv(h10_inv,trainx[1:10,],trainx[11:20,])
cat("the inverse of U10=","\n")
print(U10_inv)

#problem 3c:runtimes of the recursive and direct algorithms
#######################

# iterative method
ite_runtime=function(h10_inv,traindata,newdata){
d=dim(newdata)[1]
runtime=rep(0,d)
start_time=proc.time() # recording runtime starts
    for (i in 1:d){
      h11_inv=Hn_inv(h10_inv,traindata,newdata[i,])
      row1=h11_inv[1,]   # first row of h11_inv
      row10=h11_inv[-1,] # last ten rows of h11_inv
      e=row1[1]   # the constant e in block of Hn
      ft=row1[-1] # the transpose of f vector
      f=row10[,1] # f vector
      D=row10[,-1] # D matrix
      U10_inv=D-(1/e)*f%*%t(ft) # inverse of U10 in eq(9)
      # remove the oldest one and add a new one
      traindata=rbind(traindata[-1,],newdata[i,]) 
      h10_inv=U10_inv
      runtime[i]=proc.time()-start_time
      cat(i,"iterations","runtime=",runtime[i],"seconds","\n")
    }
    return(runtime)
}
h10_inv=hn_inv # the inverse of H10 at first time
runtime1=ite_runtime(h10_inv,trainx[1:10,],trainx[11:20,])

#training accuracy and test accuracy in iterative method
hn_inv=U_inv(h10_inv,trainx[1:10,],trainx[11:20,])
alpha_n=alpha1(hn_inv,trainx[11:20,]) # iterative alpha_n

ap3=ap(alpha_n,trainx[11:20,])

r2=R2(alpha_n,trainx[11:20,],ap3)           #  trained radius R2 

#compute the training accuracy in iterative method
ta=0
for (i in 1:dim(trainx[11:20,])[1]){ 
  dz=DZ(alpha_n,trainx[11:20,],trainx[11:20,][i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(trainx[11:20,])[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in iterative method
tt=0
for (i in 1:dim(testx)[1]){ 
  dz=DZ(alpha_n,trainx[11:20,],testx[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(testx)[1]-tt)/dim(testx)[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")

#direct method
N=10  # assume we add 10 observations sequentially
runtime2=rep(0,N)    # record the runtime for direct method
start_time2 = proc.time()
for (i in 1:N){
  hn_inv=solve(Hn(trainx[1:(i+10),]))
  runtime2[i]=proc.time()-start_time2
  cat(i,"iterations","runtime=",runtime2[i],"seconds","\n")
}

#compute the training accuracy in direct method

alpha_n=alpha1(hn_inv,trainx) # iterative alpha_n
ap3=ap(alpha_n,trainx)

r2=R2(alpha_n,trainx,ap3)           #  trained radius R2 
ta=0
for (i in 1:dim(trainx)[1]){ 
  dz=DZ(alpha_n,trainx,trainx[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(trainx)[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in direct method
tt=0
for (i in 1:dim(testx)[1]){ 
  dz=DZ(alpha_n,trainx,testx[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(testx)[1]-tt)/dim(testx)[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")

#plot the runtimes of two methods
x <- rep(1:N)
plot(x,runtime2,type="l",col="blue",xlab="the number of iterations",
     ylab="Runtime (s)", main="Runtime of the direct and iterative methods")
#Add more data to the plot
lines(x,runtime1,col="red")  #add runtimes of iterative method
legend(x=1,y=7,c("direct method","iterative method"),cex=1., 
       col=c("blue","red"),lty=c(1,1)) #add legend



#problem 4#######################
#######################

#problem 4a
#######################

#read the datafile
group1=read.delim(file.choose(),header=FALSE, sep=",")
group1_data=matrix(0,dim(group1)[1],dim(group1)[2])
# transform categorical variables into dummy variables
for (i in 1:dim(group1_data)[1]){
  for (j in 1:dim(group1_data)[2]){
  if (group1[i,j]=='x'){
    group1_data[i,j]=as.numeric(group1[i,j])} #default level x=3,
  if (group1[i,j]=='o'){
    group1_data[i,j]=as.numeric(group1[i,j])} #default level o=2
  if (group1[i,j]=='b'){
    group1_data[i,j]=as.numeric(group1[i,j])} #default level b=1
  else {group1_data[i,j]=as.numeric(group1[i,j])} #default positive=2,negative=1
}
}
# separate into positive and negative datasets               
data_pos=group1_data[group1_data[,10]==2,]  # 626 samples,default positive=2
data_neg=group1_data[group1_data[,10]==1,]  # 332 samples,default negative=1
data_pos_x=data_pos[,-10] #only contain x in class positive
data_neg_x=data_neg[,-10] #only contain x in class negative
data_pos_x=as.data.frame(data_pos_x) # dataframe type is good for computing
data_neg_x=as.data.frame(data_neg_x)

# kernel function
k=function(xi,xj,sigma){ 
  #three input
  exp(-(dist(rbind(xi,xj))[1])^2/sigma^2)
}
#define a function to update the block inverse matrix
HNK_inv2=function(hn_inv,data,newdata){
  # input Hn, n observations and new K observations 
N=dim(data)[1]
K=dim(newdata)[1]
B=matrix(0,K,N)  # matrix B: K*N
D=matrix(0,K,K)  # matrix D: K*K
I=diag(K) #identity matrix

for (i in 1:K){
  for (j in 1:N){
  B[i,j]=k(newdata[i,],data[j,],sigma)}
}
for (i in 1:K){
  for (j in 1:K){
  D[i,j]=k(newdata[i,],newdata[j,],sigma)}
}
D=D+(1/(2*C))*I
hn_inv=as.matrix(hn_inv)
w1=hn_inv-hn_inv%*%t(B)%*%solve(-D+B %*% hn_inv %*% t(B)) %*% B %*% hn_inv
w2=hn_inv %*% t(B)%*%solve(B%*%hn_inv%*%t(B)-D)
w3=solve(B%*%hn_inv%*%t(B)-D)%*% B %*% hn_inv
w4=solve(D-B%*%hn_inv%*%t(B))
hnk_inv=rbind(cbind(w1,w2),cbind(w3,w4))   
return(hnk_inv)
}

#problem 4b: check Hn+k  using the direct method
#######################

#if K=1, the results should be same as the inverse of Hn in recursive method
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel
h1=k(data_pos_x[1,],data_pos_x[1,],sigma)+1/(2*C)
h1_inv=solve(as.matrix(h1)) # the inverse of H1

#the iterative block method
hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
n=5
#a loop for computing the inverse of  Hn+1, K=1
for (i in 1:n){
# compute Hn by the iterative block method, K=1
cat("K=1,n=",i,",","inverse","of","H",i,"=","\n")
print(hn_inv)
hnk_inv=HNK_inv2(hn_inv,data_pos_x[1:i,],data_pos_x[(i+1),])
hn_inv=hnk_inv
}

hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
n=5
for (i in 1:n){
#compute alpha_n by the iterative block method, K=1
  alpha_n=alpha1(hn_inv,data_pos_x[1:i,])
  hnk_inv=HNK_inv2(hn_inv,data_pos_x[1:i,],data_pos_x[(i+1),])
  hn_inv=hnk_inv
cat("n=",i,",","alpha",i,"=",alpha_n,"\n")
}

#the direct method     
#define a function of computing Hn directly
Hn=function(data){
  d=dim(data)[1] # the number of observation,data is dataframe type
  Km=matrix(0,d,d) # K matrix named as Km
  I=diag(d) # d dimensions identity matrix
  for (i in 1:d){
    for (j in 1:d) {
      Km[i,j]=k(data[i,],data[j,],sigma)}
  }
  hn=Km+(1/(2*C))*I # compute the matrix Hn
  return (hn)
}

N=5  # assume we have 5 observations
for (i in 1:N){
  #compute Hn by direct method
  hnk_inv=solve(Hn(data_pos_x[1:i,]))
  cat("K=1,n=",i,",","inverse","of","H",i,"=","\n")
  print(hnk_inv)
}

for (i in 1:n){
  # compute alpha_n by direct method
  hnk_inv=solve(Hn(data_pos_x[1:i,]))
  alpha_n=alpha1(hnk_inv,data_pos_x[1:i,])
  cat("n=",i,",","alpha",i,"=",alpha_n,"\n")
}

#problem 4c:the accuracy and runtimes of training and test 
#######################

#define R2
R2=function(alpha_n,train,ap3){ 
  # R2 consists of three average parts:ap1,ap2,ap3 
  d=dim(train)[1]
  p1=rep(0,d)
  p2=matrix(0,d,d)
  for (i in 1:d){
    p1[i]=k(train[i,],train[i,],sigma)}
  ap1=sum(p1)/d
  
  for (i in 1:d){
    for (j in 1:d){
      p2[i,j]=k(train[i,],train[j,],sigma) }
  }
  ap2=sum(p2 %*% as.matrix(alpha_n))/d
  
  r2=ap1-2*ap2+ap3
  return(r2)
}
#define DZ
DZ=function(alpha_n,train,test,ap3){ 
  # assume dz consists of three parts:ap1,ap2,ap3
  d=dim(train)[1]
  ap1=k(test,test,sigma)
  p2=rep(0,d)
  for (i in 1:d) {
    p2[i]=k(test,train[i,],sigma)}
  
  ap2=(t(alpha_n) %*% p2)[1]
  dz=ap1-2*ap2+ap3
  return(dz) 
}
# the third term in dz and R2 is a constant ap3
#define the ap3 function
ap=function(alpha_n,train){
  # the third term in dz/R2 is a constant
  d=dim(train)[1]
  p3=matrix(0,d,d) 
  for (i in 1:d){
    for (j in 1:d){
      p3[i,j]=k(train[i,],train[j,],sigma) }
  }
  ap3=sum(as.matrix(alpha_n %*% t(alpha_n)) %*% p3)  # vectorization, not loop
  return (ap3)
}
alpha1=function(hn_inv,data){ 
  # two inputs: the inverse of Hn and n observations
  d=dim(data)[1]   # the number of observation
  e=rep(1,d) # n*1 column vector
  kj=rep(0,d) # n*1, kn vector
  for(i in 1:d){
    kj[i]=k(data[i,],data[i,],sigma)
  }
  p1=2-as.matrix(t(e)) %*% hn_inv %*% as.matrix(kj)
  p2=as.matrix(t(e)) %*% hn_inv %*% as.matrix(e)
  alpha=0.5*hn_inv %*% as.matrix((kj + (p1[1]/p2[1])*e))
  return (alpha)
}
HNK_inv2=function(hn_inv,data,newdata){
  # input Hn, n observations and new K observations 
  N=dim(data)[1]
  K=dim(newdata)[1]
  B=matrix(0,K,N)  # matrix B: K*N
  D=matrix(0,K,K)  # matrix D: K*K
  I=diag(K) #identity matrix
  
  for (i in 1:K){
    for (j in 1:N){
      B[i,j]=k(newdata[i,],data[j,],sigma)}
  }
  for (i in 1:K){
    for (j in 1:K){
      D[i,j]=k(newdata[i,],newdata[j,],sigma)}
  }
  D=D+(1/(2*C))*I
  hn_inv=as.matrix(hn_inv)
  w1=hn_inv-hn_inv%*%t(B)%*%solve(-D+B %*% hn_inv %*% t(B)) %*% B %*% hn_inv
  w2=hn_inv %*% t(B)%*%solve(B%*%hn_inv%*%t(B)-D)
  w3=solve(B%*%hn_inv%*%t(B)-D)%*% B %*% hn_inv
  w4=solve(D-B%*%hn_inv%*%t(B))
  hnk_inv=rbind(cbind(w1,w2),cbind(w3,w4))   
  return(hnk_inv)
}

#normalize data into (-1,1)
#if x==3,then x=3-2=1;
#if o==2,then o=2-2=0;
#if b==1,then b=1-2=-1
data_pos_x=data_pos_x-2 
data_neg_x=data_neg_x-2

# split datasets into training and test
set.seed(10)
index1=sample(c(1:dim(data_pos_x)[1]),106)
train=data_pos_x[index1,] #  for training
index2=sample(c(1:dim(data_neg_x)[1]),31)
test=data_neg_x[index2,] #  for test             
              
h1=k(train[1,],train[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
kk=1
tn2=round((dim(train)[1]-1)/kk)  # the number of iterations
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel   
library(tictoc)
tic()
#training starts at K=1,N=1
start_time6=proc.time() 
for (i in 1:tn2){ 
  d1=dim(hn_inv)[1]
  hnk_inv=HNK_inv2(hn_inv,train[1:d1,],train[(d1+1):(d1+kk),])
  hn_inv=hnk_inv
  cat("the number of iteration=",i,"time=",(proc.time()-start_time6),"seconds","\n")
}

alpha_n=alpha1(hn_inv,train) # iterative alpha_n
cat('Time now=',proc.time()-start_time6,"seconds","\n")
ap3=ap(alpha_n,train)
cat('Time now=',proc.time()-start_time6,"seconds","\n")
r2=R2(alpha_n,train,ap3)           #  trained radius R2 
cat("R2=",r2,"\n","the sum of alpha",sum(alpha_n),"\n",
    "K=1, Training time (s):",proc.time()-start_time6,"\n")   
toc()
#compute the training accuracy when K=1
ta=0
for (i in 1:dim(train)[1]){ 
  dz=DZ(alpha_n,train,train[i,],ap3)
if (dz<=r2){
  ta=ta+1
cat("n=",i,"It's a target","\n")}
else{
  cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(train)[1]
cat("K=1, Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy when K=1
tt=0
for (i in 1:dim(test)[1]){ 
  dz=DZ(alpha_n,train,test[i,],ap3)
if (dz<=r2){
  tt=tt+1
  cat("n=",i,"It's a target","\n")}
else{
  cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(test)[1]-tt)/dim(test)[1]
cat("K=1, Testing accuracy(%):",test_accuracy*100,"\n")

# if K=15


hn_inv=h1_inv    # the inverse of Hn starts from the inverse of H1
kk=15
tn2=round((dim(train)[1]-1)/kk)  # the number of iterations
C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel    

tic()
#training starts at K=15,N=1
start_time6=proc.time() 
for (i in 1:tn2){ 
  d1=dim(hn_inv)[1]
  hnk_inv=HNK_inv2(hn_inv,train[1:d1,],train[(d1+1):(d1+kk),])
  hn_inv=hnk_inv
  cat("the number of iteration=",i,"time=",(proc.time()-start_time6),"seconds","\n")
}

alpha_n=alpha1(hn_inv,train) # iterative alpha_n
cat('Time now=',proc.time()-start_time6,"seconds","\n")
ap3=ap(alpha_n,train)
cat('Time now=',proc.time()-start_time6,"seconds","\n")
r2=R2(alpha_n,train,ap3)           #  trained radius R2 
cat("R2=",r2,"\n","the sum of alpha",sum(alpha_n),"\n",
    "K=15, Training time (s):",proc.time()-start_time6,"\n")   
toc()
#compute the training accuracy when K=15
ta=0
for (i in 1:dim(train)[1]){ 
  dz=DZ(alpha_n,train,train[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(train)[1]
cat("K=15, Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy when K=1
tt=0
for (i in 1:dim(test)[1]){ 
  dz=DZ(alpha_n,train,test[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(test)[1]-tt)/dim(test)[1]
cat("K=15, Testing accuracy(%):",test_accuracy*100,"\n")

#problem 5#######################
#######################

#problem 5a: obtain the inverse of H 5
#######################

C=5  # the penalty factor
sigma=1 # the parameter of Gaussian Kernel 
h1=k(trainx[1,],trainx[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
h5_inv=HNK_inv2(h1_inv,trainx[1,],trainx[2:5,])#add 4 new samples
cat("K=4,","inverse","of","H5=","\n")
print(h5_inv)
  
#problem 5b:add k and remove h ssmples
#######################

#define a function to update the inverse of Hn-k or D in eq(12)
D_inv=function(h5_inv,traindata,newdata){
  kk=5 # each adding 5 samples
  hh=3 # each remove 3 samples
  d=round(dim(newdata)[1]/kk) # the number of iterations
  for (i in 1:d){
    hnk_inv=HNK_inv2(h5_inv,traindata,newdata[((i-1)*kk+1):(kk*i),])
    row_h=hnk_inv[1:hh,]   # h rows of hnk_inv
    row_n_h=hnk_inv[-(1:hh),] # last n-h rows of hnk_inv
    u11=row_h[,1:hh]   # U11
    u12=row_h[,-(1:hh)] # U12
    u12_t=row_n_h[,1:hh] # t(U12)
    un_h=row_n_h[,-(1:hh)] # D matrix
    hn_h_inv=un_h-u12_t %*% solve(u11) %*% u12 # inverse of U10 in eq(12)
    # remove the oldest h and add new k ones
    traindata=rbind(traindata[-(1:hh),],newdata[((i-1)*kk+1):(kk*i),]) 
    h5_inv=hn_h_inv
  }
  return(h5_inv)
}

#if we only use the first 20 training samples from class 1
h5_inv=HNK_inv2(h1_inv,trainx[1,],trainx[2:5,])#add 4 new samples
H11_inv=D_inv(h5_inv,trainx[1:5,],trainx[6:20,])
cat("the inverse of H11=","\n")
print(H11_inv)
#if we use all data from class 1 and -1
traind=rbind(trainx,testx)  # 30 samples in total
h5_inv=HNK_inv2(h1_inv,trainx[1,],trainx[2:5,])#add 4 new samples
H15_inv=D_inv(h5_inv,traind[1:5,],traind[6:30,])
cat("the inverse of H15=","\n")
print(H15_inv)

#problem 5c:runtimes of the recursive and direct algorithms
#######################

# iterative method
ite_runtime=function(h5_inv,traindata,newdata){
  kk=5 # each adding 5 samples
  hh=3 # each remove 3 samples
  d=round(dim(newdata)[1]/kk) # the number of iterations=5 if use 30 samples
  runtime=rep(0,d)
  start_time=proc.time() # recording runtime starts
  for (i in 1:d){
    hnk_inv=HNK_inv2(h5_inv,traindata,newdata[((i-1)*kk+1):(kk*i),])
    row_h=hnk_inv[1:hh,]   # h rows of hnk_inv
    row_n_h=hnk_inv[-(1:hh),] # last n-h rows of hnk_inv
    u11=row_h[,1:hh]   # U11
    u12=row_h[,-(1:hh)] # U12
    u12_t=row_n_h[,1:hh] # t(U12)
    un_h=row_n_h[,-(1:hh)] # D matrix
    hn_h_inv=un_h-u12_t %*% solve(u11) %*% u12 # inverse of U10 in eq(12)
    # remove the oldest h and add new k ones
    traindata=rbind(traindata[-(1:hh),],newdata[((i-1)*kk+1):(kk*i),]) 
    h5_inv=hn_h_inv

    runtime[i]=proc.time()-start_time
    cat(i,"iterations","runtime=",runtime[i],"seconds","\n")
  }
  return(runtime)
}
h1=k(trainx[1,],trainx[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
h5_inv=HNK_inv2(h1_inv,trainx[1,],trainx[2:5,])#add 4 new samples
runtime1=ite_runtime(h5_inv,traind[1:5,],traind[6:30,])
#direct method

N=5  # the number of iterative
kk=5 # each adding 5 samples
runtime2=rep(0,N)    # record the runtime for direct method
start_time2 = proc.time()

for (i in 1:N){
  hn_inv=solve(Hn(traind[1:(kk*i),]))
  runtime2[i]=proc.time()-start_time2
  cat(i,"iterations","runtime=",runtime2[i],"seconds","\n")
}

#plot the runtimes of two methods
x <- rep(1:N)
plot(x,runtime2,type="l",col="blue",xlab="the number of iterations",
     ylab="Runtime (s)", main="Runtime of the direct and iterative methods")
#Add more data to the plot
lines(x,runtime1,col="red")  #add runtimes of iterative method
legend(x=1,y=3.5,c("direct method","iterative method"),cex=1., 
       col=c("blue","red"),lty=c(1,1)) #add legend


#compute the training accuracy in iterative method

#if we only use the first 20 training samples from class 1
H11_inv=D_inv(h5_inv,trainx[1:5,],trainx[6:20,]) # get inv(H11) by iterative way

alpha_n=alpha1(H11_inv,trainx[10:20,]) # iterative alpha_n
ap3=ap(alpha_n,trainx[10:20,])       # the constant term in R2 and dz
r2=R2(alpha_n,trainx[10:20,],ap3)     #  trained radius R2 

ta=0 # set the counter of prediction 
for (i in 1:dim(trainx[10:20,])[1]){ 
  dz=DZ(alpha_n,trainx[10:20,],(trainx[10:20,])[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(trainx[10:20,])[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in iterative method
tt=0  # set the counter of prediction
for (i in 1:dim(testx)[1]){ 
  dz=DZ(alpha_n,trainx[10:20,],testx[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(testx)[1]-tt)/dim(testx)[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")

##compute the training accuracy in direct method

hn_inv=solve(Hn(traind[1:20,]))
alpha_n=alpha1(hn_inv,trainx) # iterative alpha_n
ap3=ap(alpha_n,trainx)

r2=R2(alpha_n,trainx,ap3)           #  trained radius R2 
ta=0
for (i in 1:dim(trainx)[1]){ 
  dz=DZ(alpha_n,trainx,trainx[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(trainx)[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in direct method
tt=0
for (i in 1:dim(testx)[1]){ 
  dz=DZ(alpha_n,trainx,testx[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(testx)[1]-tt)/dim(testx)[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")


#problem 5d:repeat c but use group1 data
#######################

# iterative method
ite_runtime=function(h15_inv,traindata,newdata){
  kk=15 # each adding 15 samples
  hh=10 # each remove 10 samples
  d=round(dim(newdata)[1]/kk) # the number of iterations=5 if use 30 samples
  runtime=rep(0,d)
  
  start_time=proc.time() # recording runtime starts
  for (i in 1:d){
    hnk_inv=HNK_inv2(h15_inv,traindata,newdata[((i-1)*kk+1):(kk*i),])
    row_h=hnk_inv[1:hh,]   # h rows of hnk_inv
    row_n_h=hnk_inv[-(1:hh),] # last n-h rows of hnk_inv
    u11=row_h[,1:hh]   # U11
    u12=row_h[,-(1:hh)] # U12
    u12_t=row_n_h[,1:hh] # t(U12)
    un_h=row_n_h[,-(1:hh)] # D matrix
    hn_h_inv=un_h-u12_t %*% solve(u11) %*% u12 # inverse of Un-h in eq(12)
    # remove the oldest h and add new k ones
    traindata=rbind(traindata[-(1:hh),],newdata[((i-1)*kk+1):(kk*i),]) 
    h15_inv=hn_h_inv

    runtime[i]=proc.time() - start_time
    cat(i,"iterations","runtime=",runtime[i],"seconds","\n")
  }
  return(runtime)
}

h1=k(data_pos_x[1,],data_pos_x[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
h15_inv=HNK_inv2(h1_inv,data_pos_x[1,],data_pos_x[2:15,])
runtime1=ite_runtime(h15_inv,data_pos_x[1:15,],data_pos_x[16:165,])

#direct method
kk=15 # each adding 5 samples
N=10  # the number of iterative=(165-16+1)/kk=10
runtime2=rep(0,N)    # record the runtime for direct method
start_time2 = proc.time()

for (i in 1:N){
  hn_inv=solve(Hn(traind[1:(kk*i),]))
  runtime2[i]=proc.time()-start_time2
  cat(i,"iterations","runtime=",runtime2[i],"seconds","\n")
}
#plot the runtimes of two methods
x <- rep(1:N)
plot(x,runtime2,type="l",col="blue",xlab="the number of iterations",
     ylab="Runtime (s)", main="Runtime of the direct and iterative methods")
#Add more data to the plot
lines(x,runtime1,col="red")  #add runtimes of iterative method
legend(x=1,y=200,c("direct method","iterative method"),cex=1., 
       col=c("blue","red"),lty=c(1,1)) #add legend

#compute the training accuracy in iterative method
#define a function to update the inverse of Hn-k or D in eq(12)
D_inv1=function(h5_inv,traindata,newdata){
  kk=15 # each adding 5 samples
  hh=10 # each remove 3 samples
  d=round(dim(newdata)[1]/kk) # the number of iterations
  for (i in 1:d){
    hnk_inv=HNK_inv2(h5_inv,traindata,newdata[((i-1)*kk+1):(kk*i),])
    row_h=hnk_inv[1:hh,]   # h rows of hnk_inv
    row_n_h=hnk_inv[-(1:hh),] # last n-h rows of hnk_inv
    u11=row_h[,1:hh]   # U11
    u12=row_h[,-(1:hh)] # U12
    u12_t=row_n_h[,1:hh] # t(U12)
    un_h=row_n_h[,-(1:hh)] # D matrix
    hn_h_inv=un_h-u12_t %*% solve(u11) %*% u12 # inverse of U10 in eq(12)
    # remove the oldest h and add new k ones
    traindata=rbind(traindata[-(1:hh),],newdata[((i-1)*kk+1):(kk*i),]) 
    h5_inv=hn_h_inv
  }
  return(h5_inv)
}
h1=k(data_pos_x[1,],data_pos_x[1,],sigma)+1/(2*C)
h1_inv=solve(h1) # the inverse of H1
h15_inv=HNK_inv2(h1_inv,data_pos_x[1,],data_pos_x[2:15,])
hnk_inv=D_inv1(h15_inv,data_pos_x[1:15,],data_pos_x[16:165,]) # get inv(H65) by iterative way


alpha_n=alpha1(hnk_inv,data_pos_x[101:165,]) # iterative alpha_n
ap3=ap(alpha_n,data_pos_x[101:165,])       # the constant term in R2 and dz
r2=R2(alpha_n,data_pos_x[101:165,],ap3)     #  trained radius R2 

ta=0 # set the counter of prediction 
for (i in 1:dim(data_pos_x[101:165,])[1]){ 
  dz=DZ(alpha_n,data_pos_x[101:165,],(data_pos_x[101:165,])[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(data_pos_x[101:165,])[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in iterative method
tt=0  # set the counter of prediction
for (i in 1:dim(data_neg_x[1:31,])[1]){ 
  dz=DZ(alpha_n,data_pos_x[101:165,],(data_neg_x[1:31,])[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(data_neg_x[1:31,])[1]-tt)/dim(data_neg_x[1:31,])[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")

##compute the training accuracy in direct method

hn_inv=solve(Hn(data_pos_x[1:165,]))
alpha_n=alpha1(hn_inv,data_pos_x[1:165,]) # iterative alpha_n
ap3=ap(alpha_n,data_pos_x[1:165,]) # the constant term in R2 and dz

r2=R2(alpha_n,data_pos_x[1:165,],ap3)  #  trained radius R2 
ta=0 # set the counter of prediction
for (i in 1:dim(data_pos_x[1:165,])[1]){ 
  dz=DZ(alpha_n,data_pos_x[1:165,],(data_pos_x[1:165,])[i,],ap3)
  if (dz<=r2){
    ta=ta+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
train_accuracy=ta/dim(data_pos_x[1:165,])[1]
cat(" Training accuracy(%):",train_accuracy*100,"\n")

# compute the test accuracy in direct method
tt=0 # set the counter of prediction
for (i in 1:dim(data_neg_x[1:31,])[1]){ 
  dz=DZ(alpha_n,data_pos_x[1:165,],(data_neg_x[1:31,])[i,],ap3)
  if (dz<=r2){
    tt=tt+1
    cat("n=",i,"It's a target","\n")}
  else{
    cat("n=",i,"It's an outlier","\n")}
}
test_accuracy=(dim(data_neg_x[1:31,])[1]-tt)/dim(data_neg_x[1:31,])[1]
cat("Testing accuracy(%):",test_accuracy*100,"\n")
