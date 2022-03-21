%% Download Data for US
%
% Note: WSHOMCB is the continuation of MBST
%
% Author:       Gavin Goy
% Date:         26-11-2019
% Last update:  03-12-2020

clear;
clc;
addpath('sources');


%% [0] Choices
strt = '01-Jan-1973'; % Note: you loose one observation due to growth rates
ennd = '01-Oct-2019';
sve = 1;


%% [1] Load data (order very important!!!)
% Specify series names and horizon (quarterly; monthly; weekly)
series = {'GDP','GDPC1','GDPDEF','GDPCTPI','PCEC','FPI','GPDI','COMPNFB','A006RD3Q086SBEA','BOGZ1FL792090005Q','BOGZ1FL794190005Q','FBTFASQ027S','PRS85006023',...
          'CE16OV','CNP16OV','PCEPI','FEDFUNDS','LOANINV','BAA10YM','AWHNONAG','PCEDG',... 
          'TLAACBW027SBOG','TLBACBW027SBOG','DPSACBW027SBOG','TOTBKCR','TOTCI','TREAST','WSHOMCB','WCPFF','WACBS','FEDDT','TERAUCT','OTHLT','RESPPANWW','TASACBW027SBOG','TOTLL','CLSACBW027SBOG','RELACBW027SBOG',...
          'DGS10'};
      
window={'07/01/1950','06/25/2020'};
iQ = 13;    % # of quarterly series
iM = 8;     % # of monthly series
iW = 17;    % # of weekly series
iD = 1;     % # of daily series

% Conection with FRED.
conection=fred('https://fred.stlouisfed.org/');
nseries=length(series);%We count the number of series.
for i=1:nseries
    get = fetch(conection,series(i),window(1),window(2));
    eval(['date' char(string(i)) '=datetime(datestr(get.Data(:,1)));'])
    eval([ char(series(:,i)) '= timetable(date' char(string(i)) ',get.Data(:,2));']);
    eval(['clear date'  char(string(i))])
end
close(conection)
clear i get nseries conection window


%% [2] Create time tables and quarterly averages of weekly/monthly data
% Quarterly series
quarterly = synchronize(GDP,GDPC1);
quarterly = synchronize(quarterly,GDPDEF);
quarterly = synchronize(quarterly,GDPCTPI);
quarterly = synchronize(quarterly,PCEC);
quarterly = synchronize(quarterly,FPI);
quarterly = synchronize(quarterly,GPDI);
quarterly = synchronize(quarterly,COMPNFB);
quarterly = synchronize(quarterly,A006RD3Q086SBEA);
quarterly = synchronize(quarterly,BOGZ1FL792090005Q);
quarterly = synchronize(quarterly,BOGZ1FL794190005Q);
quarterly = synchronize(quarterly,FBTFASQ027S);
quarterly = synchronize(quarterly,PRS85006023);
for i = 1:iQ
    quarterly.Properties.VariableNames{i} = char(series(1,i));
end
clear i GDP GDPC1 GDPCTPI GDPDEF PCEC FPI GPDI COMPNFB PRS85006023 BOGZ1FL792090005Q BOGZ1FL794190005Q FBTFASQ027S A006RD3Q086SBEA


% Monthly series
monthly = synchronize(CE16OV,CNP16OV);
monthly = synchronize(monthly,PCEPI);
monthly = synchronize(monthly,FEDFUNDS);
monthly = synchronize(monthly,LOANINV);
monthly = synchronize(monthly,BAA10YM);
monthly = synchronize(monthly,AWHNONAG);
monthly = synchronize(monthly,PCEDG);
for i = 1:iM
    monthly.Properties.VariableNames{i} = char(series(1,i+iQ));
end
clear AWHNONAG CE16OV CNP16OV PCEPI FEDFUNDS LOANINV BAA10YM PCEDG i 
monthly = rmmissing(retime(monthly,'quarterly','mean'));


% Weekly series
weekly = synchronize(TLAACBW027SBOG,TLBACBW027SBOG);
weekly = synchronize(weekly,DPSACBW027SBOG);
weekly = synchronize(weekly,TOTBKCR);
weekly = synchronize(weekly,TOTCI);
weekly = synchronize(weekly,TREAST);
weekly = synchronize(weekly,WSHOMCB);
weekly = synchronize(weekly,WCPFF);
weekly = synchronize(weekly,WACBS);
weekly = synchronize(weekly,FEDDT);
weekly = synchronize(weekly,TERAUCT);
weekly = synchronize(weekly,OTHLT);
weekly = synchronize(weekly,RESPPANWW);
weekly = synchronize(weekly,TASACBW027SBOG);
weekly = synchronize(weekly,TOTLL);
weekly = synchronize(weekly,CLSACBW027SBOG);
weekly = synchronize(weekly,RELACBW027SBOG);
for i = 1:iW
    weekly.Properties.VariableNames{i} = char(series(1,i+iQ+iM));
end
weekly = fillmissing(weekly,'constant',0,'DataVariables',@isnumeric);

clear TOTBKCR TOTCI TREAST WSHOMCB WCPFF WACBS FEDDT TERAUCT OTHLT RESPPANWW  ...
    TLAACBW027SBOG TLBACBW027SBOG DPSACBW027SBOG TASACBW027SBOG TOTLL CLSACBW027SBOG RELACBW027SBOG i
WeInQu = rmmissing(retime(weekly,'quarterly','mean')); 


% Daily series
daily = rmmissing(retime(DGS10,'quarterly','lastvalue'));
daily.Properties.VariableNames = {'DGS10'};
clear DGS10

% syncrhonize and cut the sample
data = synchronize(quarterly,monthly);
data = synchronize(data,WeInQu);
data = synchronize(data,daily);
t1 = find(data.date1==strt);
t2 = find(data.date1==ennd);
data = data(t1:t2,:);

clear iQ iM iW iD t1 t2 daily WeInQu monthly quarterly


%% [3] Load 10-year equivalents from the New York Fed OMO reports (manually)
% For sources see links in excel file
SOMA = readtable('SOMA_data.xlsx','Sheet','10Y equiv SOMA','Range','A2:E181');
SOMA = table2timetable(SOMA);
SOMA = fillmissing(SOMA,'constant',0,'DataVariables',@isnumeric);
SOMAQ = rmmissing(retime(SOMA,'quarterly','lastvalue'));
data = synchronize(data,SOMAQ);
mNaN = isnan([data.TotalSOMA, data.TreasurySecurities, data.AgencyMBS, data.AgencyDebt]);
data.TotalSOMA(mNaN(:,1),1) = 0;
data.TreasurySecurities(mNaN(:,2),1) = 0;
data.AgencyMBS(mNaN(:,3),1) = 0;
data.AgencyDebt(mNaN(:,4),1) = 0;
clear mNaN SOMAQ


%% [4] Construct observables (in %)
% 0. Moving average of pop growth due to spurious dynamics in CNP16OV
data.CNP16OV_ma = movmean(data.CNP16OV,4); % [4 0] for trailing MA

% 1. per capita real GDP growth
GDP = diff( log(data.GDP ./ data.GDPDEF ./ data.CNP16OV_ma) )*100;

% 2. per capita real consumption growth
Cons = diff( log( (data.PCEC-data.PCEDG) ./ data.GDPDEF ./ data.CNP16OV_ma) )*100;

% 3. per capita real investment growth
Inv = diff( log( (data.GPDI+data.PCEDG) ./ data.GDPDEF ./ data.CNP16OV_ma)*100 );

% 4. Average hours worked (weekly numbers * 13 = quarterly)
Lab = log(13*data.AWHNONAG .* data.CE16OV./ data.CNP16OV_ma)*100; 
Lab = Lab(2:end,1);
Lab = Lab-mean(Lab);

% 5. Inflation (based on GDP deflator)
Infl = diff(log(data.GDPDEF))*100;

% 6. Real wage growth
Wage = diff(log(data.COMPNFB ./ data.GDPDEF))*100;

% 7. Quarterly Fed funds rate
FFR = data.FEDFUNDS ./ 4;
FFR = FFR(2:end,1);

% 8. 10-year equivalent Treasury / GDP (both in bil), in %
CB_Bonds_10Y = ( data.TreasurySecurities ./ data.GDP )*100;
CB_Bonds_10Y = CB_Bonds_10Y(2:end,1);

% 9. Fed holdings of corporate bonds (in mil while GDP in bil)
CB_Loans = ( (data.WSHOMCB + data.FEDDT) ./ data.GDP ) ./10; 
CB_Loans = CB_Loans(2:end,1);

% 10. Fed liquidity injections (see Flemings, 2012) in mil while GDP in bn.  
CBL = ( (data.WACBS + data.WCPFF + data.TERAUCT + data.OTHLT) ./ data.GDP ) ./10;
CBL = CBL(2:end,1);

% Quarterly BAA - 10Y treasury spread 
BAA = data.BAA10YM ./4;
BAA = BAA(2:end,1);

% 10-year treasury, constant maturity
R10obs = data.DGS10 ./ 4;
R10obs = R10obs(2:end,1);

% Fed holdings of treasury (in mil while GDP in bil)
CB_Bonds = ( (data.TREAST ./ data.GDP) )./10;
CB_Bonds(1:find(data.date1=="01-Oct-2002"),1) = 5.5; % set series before 2003Q1 to 5.5%
CB_Bonds = CB_Bonds(2:end,1)-5.5; 

%% [5] Load GZ-spread
% Downloaded from: https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv
num = readtable('ebp_csv.csv');
GZSpread = num.gz_spread;
dat = datetime(1973,1:size(num,1),1)';
GZSpread = timetable(dat,GZSpread);
GZSpread = rmmissing(retime(GZSpread,'quarterly','mean'));
t1 = find(GZSpread.dat==strt)+1;
t2 = find(GZSpread.dat==ennd);
GZSpread = timetable2table(GZSpread(t1:t2,:));
GZSpread = table2array(GZSpread(:,2)) ./ 4;

clear t1 t1 dat num 


%% [6] Load ACM term premium
% Downloaded from: https://www.newyorkfed.org/research/data_indicators/term_premia.html
num = readtable('ACMTermPremium.xls','Sheet','ACM Monthly');
ACMTP10 = num.ACMTP10;
dat = datetime(1961,6:size(num,1)+5,1)';
ACMTP10 = timetable(dat,ACMTP10);
ACMTP10 = rmmissing(retime(ACMTP10,'quarterly','lastvalue'));
t1 = find(ACMTP10.dat==strt)+1;
t2 = find(ACMTP10.dat==ennd);
ACMTP10 = timetable2table(ACMTP10(t1:t2,:));
ACMTP10 = table2array(ACMTP10(:,2)) ./ 4;

clear strt ennd t1 t1 dat num 


%% [7] Save data
if sve == 1
    delete('BGS_est_data.csv') 
    date = data.date1(2:end);
    BGS_data = table(date,GDP,...
                          Cons,...
                          Inv,...
                          Lab,...
                          Infl,...
                          Wage,...
                          FFR,...
                          CB_Bonds,...
                          CB_Bonds_10Y,...
                          CB_Loans,...
                          CBL,...
                          BAA,...
                          GZSpread,...
                          ACMTP10,...
                          R10obs);
    writetable(BGS_data,'BGS_est_data.csv');
end