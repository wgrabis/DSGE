function g1 = static_g1(T, y, x, params, T_flag)
% function g1 = static_g1(T, y, x, params, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T         [#temp variables by 1]  double   vector of temporary terms to be filled by function
%   y         [M_.endo_nbr by 1]      double   vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1]       double   vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1]     double   vector of parameter values in declaration order
%                                              to evaluate the model
%   T_flag    boolean                 boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   g1
%

if T_flag
    T = rbc.static_g1_tt(T, y, x, params);
end
g1 = zeros(8, 8);
g1(1,3)=params(1);
g1(1,6)=(-1);
g1(1,8)=params(2);
g1(2,7)=(-T(1));
g1(3,2)=(-params(5));
g1(3,5)=1-(1-params(5));
g1(4,1)=(-1);
g1(4,4)=1;
g1(4,5)=(-params(3));
g1(4,8)=(-(1-params(3)));
g1(5,4)=(-1);
g1(5,5)=1;
g1(5,7)=1;
g1(6,4)=(-1);
g1(6,6)=1;
g1(6,8)=1;
g1(7,2)=(-(params(5)*params(3)*T(2)/T(1)));
g1(7,3)=(-(T(2)-params(5)*params(3)*T(2)/T(1)));
g1(7,4)=T(2);
g1(8,1)=1-params(6);
if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
end
end
