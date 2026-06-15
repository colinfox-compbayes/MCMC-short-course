function A = robfem(x,alpha,c)
% A = robfem(x,alpha,c)
% Make stiffness matrix via FEM discretization for 
% differential operator -cu'' + alpha u
% over finite domain [a,b]
% 
% input:
%  x       N+1 vector of x values sorted in ascending order
%  alpha   N vector of alpha values in each interval (x_i,x_i+1)
%  c       N vector of c values in each (x_i,x_i+1)
% output:
%  A       (N+1)*(N+1) sparse matrix containing stiffness matrix

% adapted from femprec
% cf. notes of 29/9/07

npt = length(x);
A = sparse([],[],[],npt,npt,3*npt); % blank precision matrix

% set local to global node mapping for each element
nodenum = [(1:(npt-1))' (2:npt)']; % easy in this case
% calculate lengths
el_len = x(nodenum(:,2)) - x(nodenum(:,1));

nel = length(el_len);  % number of elements
ls1 = [2 1;1 2]/12;    % unscaled local stiffness matrix for u^2 term
ls2 = [1 -1; -1 1]/2;  % unscaled local stiffness matrix for (u')^2 term

% assemble stiffness matrix by elements
for elcnt = 1:nel
    le = el_len(elcnt);   % element length
    ae = alpha(elcnt);    % element alpha value
    ce = c(elcnt);        % element c value

    % u^2 and u'^2 terms
    A(nodenum(elcnt,:),nodenum(elcnt,:)) = A(nodenum(elcnt,:),nodenum(elcnt,:)) +...
        (ae*le)*ls1 + (ce/le)*ls2;
end
%  boundary terms
A(1,1) = A(1,1) + alpha(1)/2;
A(npt,npt) = A(npt,npt) + alpha(end)/2;