function f=MISA(t,x,p,signal)

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

f(1,1) = Prod_of_A* Ha(x(1),Act_of_AToA,Trd_of_AToA,Num_of_AToA) * Hr(x(2),Inh_of_BToA,Trd_of_BToA,Num_of_BToA) - Deg_of_A*x(1);
f(2,1) = Prod_of_B* Ha(x(2),Act_of_BToB,Trd_of_BToB,Num_of_BToB) * Hr(x(1),Inh_of_AToB,Trd_of_AToB,Num_of_AToB) - Deg_of_B*x(2);

end

function H=Ha(X,lambda,S,n)
H= (1- (1-lambda)*(X.^n./(S.^n+X.^n)))/lambda;
end
function H=Hr(X,lambda,S,n)
H= 1-(1-lambda)*(X.^n./(X.^n+S.^n));
end