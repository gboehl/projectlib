%% benchmarking occbin 2019
%% use Dynare 5.1
clear

import occbin.*

rng('default'); rng(1);

%N = 100;
N = 1e6;

mu = zeros(23,1);%zeros(55,1);
load('lyapunov_cov.mat');
sig = cov*10;%diag(ones(55,1)*10);
Y = mvnrnd(mu, sig, N); 
X = zeros(N,55);

pos = [4 6 8 12 20 22 23 24 25 26 27 28 31 36 43 53 54 5 9 14 19 30 33]; 
X(:,pos) = Y;

global M_ oo_

% modnam and modnamstar below choose model
modnam = 'rank';
modnamstar = 'rank_zlb';

constraint = 'r<x_bar';
constraint_relax ='rn>x_bar';

niter= 100;
nperiods = 30;

solve_one_constraint_firstcall(modnam,modnamstar, constraint, constraint_relax)
f = waitbar(0, 'busy...');

for n=1:N

    tic
    [l,k,flag,r,rs] = solve_one_constraint_nextcall(0,'e_u',nperiods,niter,X(n,:));

    Ts(n)= toc;
    Ls(n) = l;
    Ks(n) = k;
    flags(n) = flag;

    waitbar(n/N, f);

end

save nperiods30 Ls Ks flags Ts Y

display(mean(Ts))
display(mean(Ls))
display(mean(Ks))
display(sum(flags))
