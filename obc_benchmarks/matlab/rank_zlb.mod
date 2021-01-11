 //rank.yaml ---
 //Description= variant of Smets & Wouters model
 //Authors= Felix Strobel [felix.strobel@bundesbank.de] & Gregor Boehl [gboehl@uni-bonn.de]



// variables
var dc dy di dw c c_r c_h2m r rn i q k kb rk w wh l l_r l_h2m Pi y mc z u g eps_i eps_p eps_w eps_r cf cf_r cf_h2m rf i_f qf kf kbf rkf whf lf lf_r lf_h2m yf y_gap GDP Cons Inv Wage Lab Infl FFR e_p_lag e_w_lag e_p_VAR e_w_VAR;


// shocks
varexo e_g e_z e_u e_r e_i e_p e_w;


// parameters

parameters tpr_beta lamb lamb_w phi_dy h phiss i_p i_w alpha epsilon zeta_w zeta_p sig_c sig_l phi_p psi rho_g rho_u rho_z rho_p rho_i rho_w rho_r rho mu_p mu_w phi_pi phi_y L G_Y sig_z sig_g sig_i sig_r sig_p sig_w sig_u delta trend mean_l mean_Pi rho_gz elb_level lamb_p PI gamma beta betabar RK W K k_1 K_Y Y I C c_2 kappa kappa_w x_bar;
  //para_func= [lamb_p PI gamma beta betabar RK W K k_1 K_Y Y I C c_2 kappa kappa_w x_bar]

// fixed parameters
elb_level= 0.05;
lamb= 0;
epsilon = 10;
lamb_w= 1.5;
L = 0.33;

// estimated parameters initialisation
tpr_beta = 0.10942 ;
lamb = 0.0 ;
lamb_w = 1.5 ;
phi_dy = 0.15583 ;
h = 0.84245 ;
phiss = 4.85109 ;
i_p = 0.14909 ;
i_w = 0.38564 ;
alpha = 0.18568 ;
epsilon = 10.0 ;
zeta_w = 0.75252 ;
zeta_p = 0.77935 ;
sig_c = 0.9166 ;
sig_l = 2.62716 ;
phi_p = 1.45933 ;
psi = 0.79656 ;
rho_g = 0.83855 ;
rho_u = 0.85568 ;
rho_z = 0.99566 ;
rho_p = 0.8453 ;
rho_i = 0.63556 ;
rho_w = 0.22645 ;
rho_r = 0.86377 ;
rho = 0.7934 ;
mu_p = 0.7239 ;
mu_w = 0.44809 ;
phi_pi = 1.30653 ;
phi_y = 0.24005 ;
L = 0.33 ;
G_Y = 0.18 ;
sig_z = 0.36046 ;
sig_g = 0.1716 ;
sig_i = 0.48786 ;
sig_r = 0.08091 ;
sig_p = 0.10137 ;
sig_w = 0.61065 ;
sig_u = 0.67791 ;
delta = 0.025 ;
trend = 0.29331 ;
mean_l = 0.84998 ;
mean_Pi = 0.60509 ;
rho_gz = 0.64741 ;
elb_level = 0.05 ;


// derived from steady state
PI= mean_Pi/100+1;
gamma = trend/100+1;
beta= 100/(tpr_beta+100);
betabar = beta*gamma^(-sig_c);
RK= 1/betabar -(1-delta);
lamb_p= phi_p;
W = ((1/lamb_p)*alpha^alpha*(1-alpha)^(1-alpha)*RK^(-alpha))^(1/(1-alpha));
K = alpha/(1-alpha)*W/RK*L;
k_1 = (1-(1-delta)/gamma);
K_Y = phi_p*((((1-alpha)/alpha)*(RK/W)))^(alpha-1);
Y = K/(K_Y);
I = ((1-(1-delta)/gamma)*gamma)*K;
C = (1-G_Y)*Y-I;
c_2 = (1/lamb_w)*(1-alpha)/alpha*RK*K/C;
kappa = (1-zeta_p*betabar*gamma)*(1-zeta_p)/((1+i_p*betabar*gamma)*zeta_p*((phi_p-1)*epsilon+1));
kappa_w = (1-zeta_w)*(1-zeta_w*betabar*gamma)/((1+betabar*gamma)*zeta_w*((lamb_w-1)*epsilon+1));
//conster = (PI/((beta)*(gamma)^(-(sig_c)))-1)*100;
x_bar= -(PI/((beta)*(gamma)^(-(sig_c)))-1)*100 + elb_level;

model(linear);
  z = whf+alpha*(lf-kf);
  kf = whf-rkf+lf;
  kf = (1-psi)/psi*rkf+kbf(-1);
  i_f = (1/(1+betabar*gamma))*(i_f(-1)+betabar*gamma*i_f(1)+(1/(gamma^2*phiss))*qf)+eps_i;
  kbf = (1-k_1)*kbf(-1)+(k_1)*i_f+(1+betabar*gamma)*(k_1)*(gamma^2*phiss)*eps_i;
  qf = -rf-u+(RK/(RK+(1-delta)))*rkf(+1)+((1-delta)/(RK+(1-delta)))*qf(+1);
  yf = phi_p*(alpha*kf+(1-alpha)*lf+z);
  yf = g+C/Y*cf+I/Y*i_f+RK*K_Y*(1-psi)/psi*rkf;

  cf_r = (h/gamma)/(1+h/gamma)*cf_r(-1)+(1/(1+h/gamma))*cf_r(1)+((sig_c-1)*c_2/(sig_c*(1+h/gamma)))*(lf_r-lf_r(1))-(1-h/gamma)/(sig_c*(1+h/gamma))*(rf+u);
  cf_h2m = whf+lf_h2m;
  whf = 1/(1-h/gamma)*(cf_r-h/gamma*cf_r(-1))+sig_l*lf_r;
  whf = 1/(1-h/gamma)*(cf_h2m-h/gamma*cf_h2m(-1))+sig_l*lf_h2m;
  lf = lamb*lf_h2m+(1-lamb)*lf_r;
  cf = lamb*cf_h2m+(1-lamb)*cf_r;

  c_r = (h/gamma)/(1+h/gamma)*c_r(-1)+(1/(1+h/gamma))*c_r(1)+((sig_c-1)*c_2/(sig_c*(1+h/gamma)))*(l_r-l_r(1))-(1-h/gamma)/(sig_c*(1+h/gamma))*(r-Pi(+1)+u);
  c_h2m = w+l_h2m;
  wh = 1/(1-h/gamma)*(c_r-h/gamma*c_r(-1))+sig_l*l_r;
  wh = 1/(1-h/gamma)*(c_h2m-h/gamma*c_h2m(-1))+sig_l*l_h2m;
  l = lamb*l_h2m+(1-lamb)*l_r;
  c = lamb*c_h2m+(1-lamb)*c_r;

  mc = w-z+alpha*(l-k);
  k = w-rk+l;
  k = (1-psi)/psi*rk+kb(-1);
  i = (1/(1+betabar*gamma))*(i(-1)+betabar*gamma*i(1)+(1/(gamma^2*phiss))*q)+eps_i;
  w = (1/(1+betabar*gamma))*w(-1)+(betabar*gamma/(1+betabar*gamma))*w(1)+(i_w/(1+betabar*gamma))*Pi(-1)-(1+betabar*gamma*i_w)/(1+betabar*gamma)*Pi+(betabar*gamma)/(1+betabar*gamma)*Pi(1)+kappa_w*(wh-w)+eps_w;
  kb = (1-k_1)*kb(-1)+k_1*i+(1+betabar*gamma)*k_1*gamma^2*phiss*eps_i;
  q = -r+Pi(+1)-u+(RK/(RK+(1-delta)))*rk(+1)+((1-delta)/(RK+(1-delta)))*q(+1);
  y = phi_p*(alpha*k+(1-alpha)*l+z);
  y = g+C/Y*c+I/Y*i+RK*K_Y*(1-psi)/psi*rk;
  Pi = (1/(1+betabar*gamma*i_p))*(betabar*gamma*Pi(1)+i_p*Pi(-1))+kappa*mc+eps_p;
  y_gap = y-yf;
  dc = c-c(-1);
  dy = y-y(-1);
  di = i-i(-1);
  dw = w-w(-1);

  z = rho_z*z(-1)+e_z;
  u = rho_u*u(-1)+e_u;
  eps_i = rho_i*eps_i(-1)+e_i;
  eps_r = rho_r*eps_r(-1)+e_r;

  g = rho_g*g(-1)+e_g+rho_gz*e_z;
  eps_p = rho_p*eps_p(-1)+e_p-mu_p*e_p_lag(-1);
  eps_w = rho_w*eps_w(-1)+e_w-mu_w*e_w_lag(-1);
  rn = rho*rn(-1)+(1-rho)*(phi_pi*Pi+phi_y*y_gap+phi_dy*(y_gap-y_gap(-1)))+eps_r;
  r  = x_bar;       //zlb


//aux equations
  e_p_VAR = e_p;
  e_w_VAR = e_w;
  e_p_lag = e_p_VAR(-1);
  e_w_lag = e_w_VAR(-1);
 
// measurment equations
  GDP= trend + dy;
  Cons = trend + dc;
  Inv= trend + di;
  Wage = trend + dw;
  Lab= mean_l + l;
  Infl = mean_Pi + 1*(Pi);
  FFR= (PI/((beta)*(gamma)^(-(sig_c)))-1)*100 + 1*(r);

end;


options_.noprint=1;
shocks;
var e_z;
stderr sig_z;
var e_g;
stderr sig_g;
var e_i;
stderr sig_i;
var e_u;
stderr sig_u;
var e_r;
stderr sig_r;
var e_p;
stderr sig_p;
var e_w;
stderr sig_w;
end;
