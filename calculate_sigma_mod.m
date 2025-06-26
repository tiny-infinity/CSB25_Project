function [xx]=calculate_sigma(x,p,signal,kk,d)

%par is parameters
%d is diffusion coefficient
%the xx is the vector of sigma which is Row Major Order

%%lead in the parameters
Prod_of_A = p(1);
Prod_of_B = p(2);
Deg_of_A = p(3);
Deg_of_B = p(4);
Trd_of_BToA = p(5);
Num_of_BToA = p(6);
Inh_of_BToA = p(7);
Trd_of_AToA = p(8);
Num_of_AToA = p(9);
Act_of_AToA = p(10);
Trd_of_AToB = p(11);
Num_of_AToB = p(12);
Inh_of_AToB = p(13);
Trd_of_BToB = p(14);
Num_of_BToB = p(15);
Act_of_BToB = p(16);

syms A B
 Ajac=jacobian([ Prod_of_A* Ha(A,Act_of_AToA,Trd_of_AToA,Num_of_AToA) * Hr(B,Inh_of_BToA,Trd_of_BToA,Num_of_BToA) - Deg_of_A*A;
    Prod_of_B* Ha(B,Act_of_BToB,Trd_of_BToB,Num_of_BToB) * Hr(A,Inh_of_AToB,Trd_of_AToB,Num_of_AToB) - Deg_of_B*B]);
Ajac=subs(Ajac,{'A','B'},{x(1),x(2)});
Ajac=double(Ajac);

% A*sigma+sigma*A'+2D


P=zeros(kk^2,kk^2);  %coefficient matrix

%%the initial of coeffiicient matrix
for i=0:(kk-1)
    P(i*kk+1:i*kk+kk,i*kk+1:i*kk+kk)=P(i*kk+1:i*kk+kk,i*kk+1:i*kk+kk)+Ajac;
end

for m=0:kk-1
    for i=1:kk
        for j=1:kk
            P(m*kk+i,(j-1)*kk+i)=P(m*kk+i,(j-1)*kk+i)+Ajac(m+1,j);
        end
    end
end

B=zeros(kk^2,1);
for i=1:kk
    B((i-1)*kk+i)=-2*d;
end


xx=P\B;

end

function H=Ha(X,lambda,S,n)
H= (1- (1-lambda)*(X.^n./(S.^n+X.^n)))/lambda;
end
function H=Hr(X,lambda,S,n)
H= 1-(1-lambda)*(X.^n./(X.^n+S.^n));
end