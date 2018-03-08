% generate synthetic test problem

set(0,'defaultAxesFontSize',16)

n=1000;
d=2;

mu1=[0,0]; % means for GM model
mu0=[-1,0];
mu00=[1.5,1];
noise=0.5;

y=zeros(n,1); % will contain target values
X=zeros(n,d); % will contain features
for i=1:n
    if rand<0.4 % class 1 is a single Gaussian
        y(i)=1;
        X(i,:)=mu1+noise*[0.5,2].*randn(1,d);
    else
        y(i)=0;
        if rand<0.8 % class 2 is itself a GM
            X(i,:)=mu0+noise*[0.5,2].*randn(1,d);
        else
            X(i,:)=mu00+0.5*noise*randn(1,d);
        end
    end
end

I1=find(y==1);
I0=find(y==0);

% do standard logisitc regression

w=Logistic_Regression(X,y*2-1);

% scatter plot with logistic regresison decision boundary

plot(X(I1,1),X(I1,2),'x',X(I0,1),X(I0,2),'d')
xlabel('$x_{1}$','Interpreter','latex')
ylabel('$x_{2}$','Interpreter','latex')
legend('1','0')
b1=[-w(3)/w(1),0];
bt=4*[w(2),-w(1),0];
line([b1(1)-bt(1),b1(1)+bt(1)],[b1(2)-bt(2),b1(2)+bt(2)],'color','black','linewidth',2)
axis([-2 2.5 -3 3])
pause

% now optimze linear model for auc

N=100; % simple maxits no convergence check for now
auc=zeros(N,1);

% use logistic regression as starting point
% logistic parameter w(1:d)=weights, w(d+1)=bias
% so just use w(1:d) for auc copmutation as bias does not affect result

w=w(1:d);
%w=randn(d,1);
%w=w/norm(w);
U=X*w;
[auc(1),~]=AUC(U,y,1); % get auc score for logistic regression

for its=2:N
    [auc(its),dfdu]=AUC(U,y,0); % evaluate auc and get derivative wrt u
    dfdw=X'*dfdu;
    w=w+0.2*dfdw; % learning step
    w=w/norm(w); % maintain norm(w)=1
    U=X*w;
end
[~,~]=AUC(U,y,1);

% plot scatter with auc optimzed descision normal 
figure
plot(X(I1,1),X(I1,2),'x',X(I0,1),X(I0,2),'d')
line([10*w(2),-10*w(2)],[-10*w(1),10*w(1)],'color','black','linewidth',2)
xlabel('$x_{1}$','Interpreter','latex')
ylabel('$x_{2}$','Interpreter','latex')
legend('1','0')
axis([-2 2.5 -3 3])
pause

% plot auc convergence
figure
plot(auc)
xlabel('iterations','Interpreter','latex')
ylabel('auc','Interpreter','latex')