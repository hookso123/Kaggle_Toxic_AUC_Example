function w=Logistic_Regression(X,y)

% get sizes of data sets

[n_train,d]=size(X);

% add constant bias variable

X=[X,ones(n_train,1)];

% zero regularization

lam=0;

w=zeros(d+1,1); % model parameter

start_checking=10;
tol=10^-6;
converged=0;
fold=inf;
its=0;
while converged==0
    
    sig=1./(1+exp(-y.*(X*w)));
    
    f=sum(log(1+exp(-y.*(X*w))))/n_train+lam*(w(1:d)'*w(1:d));
    
    g=X'*(-y.*(1-sig))/n_train+2*lam*[w(1:d),
        0];
    
    H=X'*diag(sig.*(1-sig))/n_train*X+2*lam*[eye(d),zeros(d,1)
        zeros(1,d),0];
    H=X'*sparse(1:n_train,1:n_train,sig.*(1-sig))*X/n_train+2*lam*[eye(d),zeros(d,1)
        zeros(1,d),0];
    
    dw=H\-g;
    
    w=w+dw;
    
    its=its+1;
    if its>start_checking
        if f>fold-tol
            converged=1;
        end
    end
    fold=f;
    
end

