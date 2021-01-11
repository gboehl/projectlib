% solve_one_constraint [zdatalinear zdatapiecewise zdatass oo base M base] = solve one constraint(modnam, modnamstar, constraint, constraint relax, shockssequence, irfshock, nperiods, maxiter, init);
%
% Inputs:
% modnam: name of .mod file for the reference regime (excludes the .mod extension).
% modnamstar: name of .mod file for the alternative regime (excludes the .mod exten- sion).
% constraint: the constraint (see notes 1 and 2 below). When the condition in constraint evaluates to true, the solution switches from the reference to the alternative regime.
% constraint relax: when the condition in constraint relax evaluates to true, the solution returns to the reference regime.
% shockssequence: a sequence of unforeseen shocks under which one wants to solve the model (size T??nshocks).
% irfshock: label for innovation for IRFs, from Dynare .mod file (one or more of the ?varexo?).
% nperiods: simulation horizon (can be longer than the sequence of shocks defined in shockssequence; must be long enough to ensure convergence back to the reference model at the end of the simulation horizon and may need to be varied depending on the sequence of shocks).
% maxiter: maximum number of iterations allowed for the solution algorithm (20 if not specified).
% init: the initial position for the vector of state variables, in deviation from steady state (if not specified, the default is steady state). The ordering follows the definition order in the .mod files.
%
% Outputs:
% zdatalinear: an array containing paths for all endogenous variables ignoring the occasionally binding constraint (the linear solution), in deviation from steady state. Each column is a variable, the order is the definition order in the .mod files.
% zdatapiecewise: an array containing paths for all endogenous variables satisfying the occasionally binding constraint (the occbin/piecewise solution), in deviation from steady state. Each column is a variable, the order is the definition order in the .mod files.
% zdatass: theinitialpositionforthevectorofstatevariables,indeviationfromsteady state (if not specified, the default is a vectors of zero implying that the initial conditions coincide with the steady state). The ordering follows the definition order in the .mod files.
% oobase,Mbase: structures produced by Dynare for the reference model ? see Dynare User Guide.

% Log of changes:
% 6/17/2013 -- Luca added a trailing underscore to local variables in an
% attempt to avoid conflicts with parameter names defined in the .mod files
% to be processed.
% 6/17/2013 -- Luca replaced external .m file setss.m


function [l, k, superflag, regime, regimestart] = ...
    solve_one_constraint_nextcall(shockssequence_,irfshock_,nperiods_,maxiter_,init_)

global M_ oo_

global oobase_  Mbase_ Mstar_

global cof cofstar ...
       Jbarmat Jbarmatstar  ...
       Dbarmatstar ...
       decrulea decruleb ...
       constraint_difference_ ...
       constraint_relax_difference_

x_bar=-1.5032;

Mstar_.params = Mbase_.params;


zdatass_ = oobase_.dr.ys;
nvars_ = Mbase_.endo_nbr;

endog_ = Mbase_.endo_names;
exog_ =  Mbase_.exo_names;




nshocks_ = size(shockssequence_,1);

% if necessary, set default values for optional arguments
if ~exist('init_')
    init_ = zeros(nvars_,1);
end

if ~exist('maxiter_')
    maxiter_ = 20;
end

if ~exist('nperiods_')
    nperiods_ = 100;
end


% set some initial conditions and loop through the shocks
% period by period
init_orig_ = init_;
zdatapiecewise_ = zeros(nperiods_,nvars_);
wishlist_ = endog_;
nwishes_ = length(wishlist_);
violvecbool_ = zeros(nperiods_+1,1);
superflag = 0;

for ishock_ = 1:nshocks_

    changes_=1;
    iter_ = 0;


    while (changes_ & iter_<maxiter_)
        iter_ = iter_ +1;

        % analyze when each regime starts based on current guess
        [regime, regimestart, flag]=map_regime(violvecbool_);
        
        if flag == 1
            superflag = 1;
        end

        % get the hypothesized piece wise linear solution
        [zdatalinear_]=mkdatap_anticipated(nperiods_,decrulea,decruleb,...
                                           cof,Jbarmat,cofstar,Jbarmatstar,Dbarmatstar,...
                                           regime,regimestart,violvecbool_,...
                                           endog_,exog_,irfshock_,shockssequence_(ishock_,:),init_);

        for i_indx_=1:nwishes_
            eval([wishlist_{i_indx_},'_difference=zdatalinear_(:,i_indx_);']);
        end



        newviolvecbool_ = eval(constraint_difference_);
        relaxconstraint_ = eval(constraint_relax_difference_);



        % check if changes to the hypothesis of the duration for each
        % regime
        if (max(newviolvecbool_-violvecbool_>0)) | sum(relaxconstraint_(find(violvecbool_==1))>0)
            changes_ = 1;
        else
            changes_ = 0;
        end



        violvecbool_ = (violvecbool_|newviolvecbool_)-(relaxconstraint_ & violvecbool_);


    end

    init_ = zdatalinear_(1,:);
    zdatapiecewise_(ishock_,:)=init_;
    init_= init_';

    % reset violvecbool_ for next period's shock -- this resetting is
    % consistent with expecting no additional shocks
    violvecbool_=[violvecbool_(2:end);0];

end

init_out = init_;
% if necessary, fill in the rest of the path with the remainder of the
% last IRF computed.
zdatapiecewise_(ishock_+1:end,:)=zdatalinear_(2:nperiods_-ishock_+1,:);

% get the linear responses
zdatalinear_ = mkdata(max(nperiods_,size(shockssequence_,1)),...
                      decrulea,decruleb,endog_,exog_,...
                      wishlist_,irfshock_,shockssequence_,init_orig_);

if changes_ ==1
%     display('Did not converge -- increase maxiter_')
    superflag = 1;
end
if (length(regimestart) == 3)
    l = regimestart(2) - 1;
    k = regimestart(3) - regimestart(2);
elseif (length(regimestart) == 2)
    l = 0;
    k = regimestart(2) - 1;
else 
    k = 0;
    l = 0;
end

