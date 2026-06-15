function X=mcgaus(mu,sig,x0,N,w)
%function X=mcgaus(mu,sig,x0,N,w)
%
% Return N samples of N(mu,sig^2) using a 
% RWM with window w, starting at x0

X = zeros(1,N);
X(1)=x0;

for k=1:(N-1)
   %xp=X(k)+w*(2*rand-1); % uniform window
   xp=X(k)+w*randn; % normal window, variance w^2
   
   alpha=min(1,exp( (-(xp-mu)^2+(X(k)-mu)^2)/(2*sig^2) ));
   
   if rand<alpha
      X(k+1)=xp;     % accept
   else
      X(k+1)=X(k);   % reject
   end
end

%>> X=mcgaus(0,1,1,1000000,1);
%>> hist(X,30)
%>> var(X)
%ans =
%   0.9969
%>> mean(X)
%ans =
%    0.0011
