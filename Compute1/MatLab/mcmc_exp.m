% script file to implement MCMC for exponential distribution, unit mean

n = 1e3;   % length of Markov chain
w = 1;     % width of random-walk window (if used)
X=[];      % initialize output vector (not very efficient)
X(1)=1;    % initial state

for k=1:(n-1)
   xp = X(k)+w*(rand-1/2);
        
   if xp < 0
       alpha = 0;
   else
       %alpha = exp(-xp)/exp(-X(k)); % uncomment this line (and comment next) to see effect of round-off errors
       alpha = exp(-xp+X(k));  % compute in the logarithm to avoid roundoff errors
   end
   
   if rand<alpha
      X(k+1)=xp;
   else
      X(k+1)=X(k);
   end
end

%try commands ...
%>> hist(X,100)
