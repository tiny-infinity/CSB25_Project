from sympy import *

""" TEACHING MYSELF SymPy"""
x,y = symbols('x y')
add=x*y
y=2


integral=integrate(add,(x,-1,1))
derivative=diff(add,x)
print(add)
print(derivative)
print(integrate(derivative,x))

y=Function('y')
from sympy import *
t,x=symbols('t x')
solution_y=dsolve(Eq(y(t).diff(t,t)-y(t),exp(t)),y(t))
print(solution_y)

#generating LaTeX scripts
print(latex(Integral(cos(x),(x,0,1))))

#subsituting values
print(solution_y.subs(t,0))

##jacobian matrix
rho,phi = symbols('rho phi')
A = Matrix([rho**2,2*rho*phi,phi*2])
B = Matrix([rho,phi])
print(A.jacobian(B).subs({rho:1,phi:1})) 