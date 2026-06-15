% script file to plot IACT as a function of window size for MH MCMC (RWM) sampling from a standard normal

ws = linspace(1,8,100); %window sizes

mu = 0;
sig = 1;
x0 = 0;

n=1e5;


for cnt = 1:length(ws)
  w = ws(cnt);
  X = mcgaus(mu,sig,x0,n,w);
  [value,dvalue,ddvalue,tauint,dtauint,Qval] = UWerr(X',1.5,length(X),0);
  taus(cnt) = tauint*2; %to get stats definition of IACT
end

plot(ws,taus)
%print -deps tauvsw_norm

