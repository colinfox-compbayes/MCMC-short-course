function [D,LL] = Dmcmc(Dtrue,N)
% Function file to simulate noisy data in inverse diffusion problem
% then perform RWM MCMC to generate a chain of length L
% that targets the posterior distribution over (constant) D
%
% cf. IP notes, section 7.6

L = 10;
s = 0.03;
Tmax = 2;
xdim = 25;
x = linspace(0,L,xdim);
xDat = 2:xdim-1; % indexes of locations of observations
nxDat = length(xDat);

r1 = 0.75*L; r2 = 0.8*L;
u0 = zeros(1,xdim);
u0(find(x>=r1 & x<=r2)) = 1;

% Simulate some data
uDat = heat(Dtrue)  + s*randn(1,nxDat); % this is the fake data
%% Uncomment to plot the fake data
%figure(1); surf(x,t,usoln); hold on;
%set(plot3(x(xDat),t(end)*ones(1,nxDat),uDat,'r-o'),'LineWidth',3);
%hold off; drawnow

%Now some MCMC
XD = 1; D = zeros(1,N); D(1) = XD; 
u = heat(XD);	
oLLkd = sum(sum(-(u-uDat).^2))/(2*s^2);  % log likelihood of initial state
LL = zeros(1,N); LL(1) = oLLkd;          % store the log likelihood

w = 0.1; % RWM window
for n = 2:N
 XDp = XD + w*(2*rand-1);
 if XDp > 0
   u = heat(XDp);                        % solve PDE
   nLLkd = sum(sum(-(u-uDat).^2))/(2*s^2);       
   alpha = exp(nLLkd - oLLkd);
   if rand < alpha
     XD = XDp;
     oLLkd = nLLkd;
   end
 end
 D(n) = XD; LL(n) = oLLkd;               % store statistics
end


function uDat = heat(Ddiff)
% return simulated data for diffusivity Ddiff 
  [K,M] = heatfem(x,ones(1,xdim-1),-Ddiff*ones(1,xdim-1));
  A = inv(M)*K;
  [t,usoln] = ode45(@heatODE,[0 Tmax],u0);
  uDat = usoln(end,xDat); 



function udot = heatODE(t,u)
  udot = A*u;
  % assert zero boundary conditions
  udot(1) = 0;
  udot(end) = 0;
end

end

end
