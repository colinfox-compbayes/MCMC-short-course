% proposal window
X=mcgaus(0,1,1,1e3,.3);
figure(1),subplot(311), plot(X)
X=mcgaus(0,1,1,1e3,3);
figure(1),subplot(312), plot(X)
X=mcgaus(0,1,1,1e3,30);
figure(1),subplot(313), plot(X)

pause
% burn in

X=mcgaus(0,1,20,1e3,3);
figure(2), plot(X)

pause
% equilibrium distribution

X=mcgaus(0,1,1,1e6,3);
figure(2), hist(X,30)

pause
% some stats

var(X)
mean(X)
