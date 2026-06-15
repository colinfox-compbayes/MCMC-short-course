function [K,M] = heatfem(x,m,D)
% [K,M]  = heatfem(x,alpha,c)
% Make stiffness matrix and mass matrix via FEM discretization for 
% differential operators -Du'' and  m u
% over finite domain [a,b]
% 
% input:
%  x       N+1 vector of x values sorted in ascending order
%  m       N vector of mass values in each interval (x_i,x_i+1)
%  D       N vector of D values in each (x_i,x_i+1)
% output:
%  K       (N+1)*(N+1) stiffness matrix
%  M       (N+1)*(N+1) mass matrix

% for BUC5

npt = length(x);
K = zeros(npt,npt);     % blank stiffness matrix
M = zeros(npt,npt);     % blank mass matrix

% set local to global node mapping for each element
nodenum = [(1:(npt-1))' (2:npt)']; % easy in this case
% calculate lengths
el_len = x(nodenum(:,2)) - x(nodenum(:,1));

nel = length(el_len);  % number of elements
ls1 = [2 1;1 2]/6;     % local stiffness matrix for u^2 term
ls2 = [1 -1; -1 1];    % local stiffness matrix for (u')^2 term

% assemble matrices by elements
for elcnt = 1:nel
    le = el_len(elcnt);   % element length
    me = m(elcnt);        % element m value
    De = D(elcnt);        % element D value

    % u^2 and u'^2 terms
    K(nodenum(elcnt,:),nodenum(elcnt,:)) = K(nodenum(elcnt,:),nodenum(elcnt,:)) + (De/le)*ls2;
    M(nodenum(elcnt,:),nodenum(elcnt,:)) = M(nodenum(elcnt,:),nodenum(elcnt,:)) + (me*le)*ls1;
end

