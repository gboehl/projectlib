%


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


function solve_one_constraint_firstcall(modnam_,modnamstar_,constraint_, constraint_relax_)

global M_ oo_

global oobase_  Mbase_ Mstar_

global cof cofstar cof01 cof11 ...
       Jbarmat Jbarmatstar  ...
       Dbarmatstar ...
       decrulea decruleb ...
       constraint_difference_ ...
       constraint_relax_difference_


% solve the reference model linearly
eval(['dynare ',modnam_,' noclearall nolog '])
oobase_ = oo_;
Mbase_ = M_;

% import locally the values of parameters assigned in the reference .mod
% file
for i_indx_ = 1:Mbase_.param_nbr
    eval([Mbase_.param_names{i_indx_},'= M_.params(i_indx_);']);
end

% Create steady state values of the variables if needed for processing the constraint
for i=1:Mbase_.endo_nbr
    eval([Mbase_.endo_names{i} '_ss = oobase_.dr.ys(i); ']);
end

% parse the .mod file for the alternative regime
eval(['dynare ',modnamstar_,' noclearall nolog '])
oostar_ = oo_;
Mstar_ = M_;


% check inputs
if ~strcmp(Mbase_.endo_names,Mstar_.endo_names)
    error('The two .mod files need to have exactly the same endogenous variables declared in the same order')
end

if ~strcmp(Mbase_.exo_names,Mstar_.exo_names)
    error('The two .mod files need to have exactly the same exogenous variables declared in the same order')
end

if ~strcmp(Mbase_.param_names,Mstar_.param_names)
    warning('The parameter list does not match across .mod files')
end

% ensure that the two models have the same parameters
% use the parameters for the base model.
Mstar_.params = Mbase_.params;


zdatass_ = oobase_.dr.ys;


% get the matrices holding the first derivatives for the model
% each regime is treated separately
[hm1,h,hl1,Jbarmat] = get_deriv(Mbase_,zdatass_);
cof = [hm1,h,hl1];


[hm1,h,hl1,Jbarmatstar,resid] = get_deriv(Mstar_,zdatass_);
cofstar = [hm1,h,hl1];
Dbarmatstar = resid;



if isfield(Mbase_,'nfwrd')
    % the latest Dynare distributions have moved nstatic and nfwrd
    [decrulea,decruleb]=get_pq(oobase_.dr,Mbase_.nstatic,Mbase_.nfwrd);
else
    [decrulea,decruleb]=get_pq(oobase_.dr,oobase_.dr.nstatic,oobase_.dr.nfwrd);
end



zdatass_ = oobase_.dr.ys;
nvars_ = Mbase_.endo_nbr;

endog_ = Mbase_.endo_names;
exog_ =  Mbase_.exo_names;


% processes the constraints specified in the call to this function
% uppend a suffix to each endogenous variable
constraint_difference_ = process_constraint(constraint_,'_difference',Mbase_.endo_names,0);

constraint_relax_difference_ = process_constraint(constraint_relax_,'_difference',Mbase_.endo_names,0);

