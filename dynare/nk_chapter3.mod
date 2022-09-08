//NK model with wage stickiness -
//Chapter 4 (UNDERSTANDING DSGE MODELS)
var Y I C R K W L PIW P PI A CM;
varexo e;
parameters sigma phi alpha beta delta rhoa psi theta thetaW psiW;
sigma = 2;
phi = 1.5;
alpha = 0.35;
beta = 0.985;
delta = 0.025;
rhoa = 0.95;
psi = 8;
theta = 0.75;
thetaW = 0.75;
psiW = 21;
model(linear);
#Pss = 1;
#Rss = Pss*((1/beta)-(1-delta));
#CMss = ((psi-1)/psi)*(1-beta*theta)*Pss;
#Wss = (1-alpha)*(CMss^(1/(1-alpha)))*((alpha/Rss)^(alpha/(1-alpha)));
#Yss = ((Rss/(Rss-delta*alpha*CMss))^(sigma/(sigma+phi)))*((1-beta*thetaW)
*((psiW-1)/psiW)*(Wss/Pss)*(Wss/((1-alpha)*CMss))^phi)^(1/(sigma+phi));
#Kss = alpha*CMss*(Yss/Rss);
#Iss = delta*Kss;
#Css = Yss - Iss;
#Lss = (1-alpha)*CMss*(Yss/Wss);
//1-Phillips equation for wages
PIW = beta*PIW(+1)+((1-thetaW)*(1-beta*thetaW)/thetaW)*(sigma*C+phi*L-(W-P));
//2-Gross wage inflation rate
PIW = W - W(-1);
//3-Euler equation
(sigma/beta)*(C(+1)-C)=(Rss/Pss)*(R(+1)-P(+1));
//4-Law of motion of capital
K = (1-delta)*K(-1) + delta*I;
//5-Production function
Y = A + alpha*K(-1) + (1-alpha)*L;
//6-Demand for capital
K(-1) = Y - R;
//7-Demand for labor
L = Y - W;
//8-Marginal cost
CM = ((1-alpha)*W + alpha*R - A);
//9-Phillips equation
PI = beta*PI(+1)+((1-theta)*(1-beta*theta)/theta)*(CM-P);
//10-Gross inflation rate
PI = P - P(-1);
//11-Equilibrium condition
Yss*Y = Css*C + Iss*I;
//12-Productivity shock
A = rhoa*A(-1) + e;
end;
NK model with rigidity in households 113
model_diagnostics;
steady;
check (qz_zero_threshold=1e-20);
shocks;
var e;
stderr 0.01;
end;
stoch_simul(qz_zero_threshold=1e-20) Y I C R K W L PI A;