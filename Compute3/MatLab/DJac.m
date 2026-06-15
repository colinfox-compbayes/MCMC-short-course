function J = DJac(Dtrue)
% Function file to evaluate jacobian of forward map from D(x) to data u(x,T)
% Input: scalar Dtrue > 0 , Jacobian is evaluated at this (constant) coefficient
% Output: J is the xdim-1 x nxDat (number of FEM elements x number of internal FEM nodes)
%            secant approximation to the Jacobian 
%
% cf. 22 Sept 2016

% use the constants in sec. 7.6 of Inverse Problems notes
L = 10;
Tmax = 2;
xdim = 25;
x = linspace(0,L,xdim);
xDat = 2:xdim-1; % indexes of locations of observations
nxDat = length(xDat);

r1 = 0.75*L; r2 = 0.8*L;
u0 = zeros(1,xdim);
u0(find(x>=r1 & x<=r2)) = 1;

% Simulate noise free data for basis of perturbations 
[K,M] = heatfem(x,ones(1,xdim-1),-(Dtrue*ones(1,xdim-1)));
A = inv(M)*K;
[t,usoln] = ode45(@heatODE,[0 Tmax],u0);
uRef = usoln(end,xDat);

% standard basis ...
Dpert = eye(xdim-1,xdim-1);
scale  = 0.1; % this is my step in the secant method
J = zeros(nxDat,xdim-1); % matrix to hold Jacobian

for count = 1:xdim-1
  [K,M] = heatfem(x,ones(1,xdim-1),-(Dtrue*ones(1,xdim-1)+scale*Dpert(:,count)'));
  A = inv(M)*K;
  [t,usoln] = ode45(@heatODE,[0 Tmax],u0);
  uDat = usoln(end,xDat);
  J(:,count) = (uDat-uRef)'/(scale);
end 
 

function udot = heatODE(t,u)
  udot = A*u;
  % assert zero boundary conditions
  udot(1) = 0;
  udot(end) = 0;
end


end
