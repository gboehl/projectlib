%% benchmarking occbin 2019
clear

rng('default'); rng(1);

N = 1e6;

mu = zeros(55,1);
sig = diag(ones(55,1)*10);
X = mvnrnd(mu, sig, N); 

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

save nperiods30 Ls Ks flags Ts

display(mean(Ts))
display(mean(Ls))
display(mean(Ks))
display(sum(flags))
