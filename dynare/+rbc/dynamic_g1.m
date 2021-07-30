function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
% function g1 = dynamic_g1(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   g1
%

if T_flag
    T = rbc.dynamic_g1_tt(T, y, x, params, steady_state, it_);
end
g1 = zeros(8, 13);
g1(1,5)=params(1);
g1(1,8)=(-1);
g1(1,10)=params(2);
g1(2,5)=(-(params(1)/params(4)));
g1(2,11)=params(1)/params(4);
g1(2,12)=(-T(1));
g1(3,4)=(-params(5));
g1(3,2)=(-(1-params(5)));
g1(3,7)=1;
g1(4,3)=(-1);
g1(4,6)=1;
g1(4,2)=(-params(3));
g1(4,10)=(-(1-params(3)));
g1(5,6)=(-1);
g1(5,2)=1;
g1(5,9)=1;
g1(6,6)=(-1);
g1(6,8)=1;
g1(6,10)=1;
g1(7,4)=(-(params(5)*params(3)*T(2)/T(1)));
g1(7,5)=(-(T(2)-params(5)*params(3)*T(2)/T(1)));
g1(7,6)=T(2);
g1(8,1)=(-params(6));
g1(8,3)=1;
g1(8,13)=(-1);

end
