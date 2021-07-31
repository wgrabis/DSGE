close all;

var y R infl a;

varexo ea;

parameters b phi mu1 mu2 mu3 rho;


b       = -1;
phi     = 0.89;
mu1     = 0.5;
mu2     = 0.5;
mu3     = 0;
rho     = 0.9;

model(linear);

y                        = y(+1) + b*(R - infl);
y                        = phi*y(-1) - phi*a(-1) + a;
R                        = mu1*infl(-1) + mu2*y - mu2*a;
a                        = rho*a(-1) + ea;

end;
steady;

shocks;
var ea=0.01;
end;

check;

stoch_simul(order=1) y R infl a;   