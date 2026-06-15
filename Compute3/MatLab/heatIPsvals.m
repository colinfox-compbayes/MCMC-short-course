% Script file to plot singular values of inverse heat problem, for intial conditions
% (these are actually eigenvlues as input/output spaces are the same)
% cf notes 22 Sept 2016 (in Bath)

% Constants from geoff's example, cf. IP notes sec. 7.6

D=0.5;
L=10;
T=2;

alpha = exp(-D*T*(pi/L)^2);

l = 1:15;
semilogy(l,alpha.^(l.^2),'o','MarkerSize',10,'LineWidth',2)

set(gca,'FontSize',20)
xlabel('index'), ylabel('eigenvalue')
%print -dpng HIPevals


