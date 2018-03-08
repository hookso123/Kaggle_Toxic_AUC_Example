function [f,dfdu]=AUC(U,y,ploton)

yI1=find(y==1); % index for each class
yI0=find(y==0);

n1=length(yI1);
n0=length(yI0);

U1=sort(U(yI1)); % sort ranking variables for each class
U0=sort(U(yI0));

% calcaulte AUC score

count=0; % count number of pairs (i,j) with u1(i)>u0(j)
j=1;
for i=1:n1
    while (j<=n0 && U1(i)>U0(j))
        j=j+1;
    end
    count=count+(j-1);
end
f=count/(n1*n0);

% get U densities for each class

m=50; % approximate density by linear interpolation between m points
I1=floor(linspace(1,n1,m-1));
I0=floor(linspace(1,n0,m-1));

X1=zeros(m,1);
D1=zeros(m,1);
X1(1)=U1(I1(1))+(U1(I1(1))-U1(I1(2)))/2;
for i=2:m-1
    X1(i)=(U1(I1(i-1))+U1(I1(i)))/2;
    D1(i)=(I1(i)-I1(i-1))/(U1(I1(i))-U1(I1(i-1)));
end
X1(m)=U1(I1(m-1))+(U1(I1(m-1))-U1(I1(m-2)))/2;

X0=zeros(m,1);
D0=zeros(m,1);
X0(1)=U0(I0(1))+(U0(I0(1))-U0(I0(2)))/2;
for i=2:m-1
    X0(i)=(U0(I0(i-1))+U0(I0(i)))/2;
    D0(i)=(I0(i)-I0(i-1))/(U0(I0(i))-U0(I0(i-1)));
end
X0(m)=U0(I0(m-1))+(U0(I0(m-1))-U0(I0(m-2)))/2;

if ploton % make a nice plot of the two densities
    figure
    plot(X1,D1/n1,X0,D0/n0)
    legend('1','0')
    xlabel('$u$','Interpreter','latex')
    ylabel('$\rho_{i}(u)$','Interpreter','latex')
    pause
end

% compute df/du

dfdu=zeros(n0+n1,1);
dfdu(yI0)=-interp1q(X1,D1,U(yI0'))/(n1*n0);
dfdu(yI1)=interp1q(X0,D0,U(yI1'))/(n1*n0);
dfdu(isnan(dfdu))=0;

end
