function [ outputs ] = mcmc_function_name(nsamp)
% Template for function file implements MCMC sampling counting assignment
% using a number of 'moves' for the proposal

% Input: nsamp -- number of steps of the MCMC

% Read in image file
slide = double(imread('slide.tif'))/255; % this is taken as data. Dividing by 255 undoes the scale when writing a tif to UINT8
% plot the slide
figure(1), clf, colormap(gray(256))
image(slide,'CDataMapping','scaled')
axis('square')

MoveRatio = [1 1 1 1]; 
MoveProb = cumsum(MoveRatio(1:end-1)/sum(MoveRatio));

% any constants are declared here

X = [];                  % starting state is null state
llold = loglike(X);
lpold = logprior(X);

% initialize outputs here

% here is the MCMC         
for iter = 2:nsamp
    kernum = sum(MoveProb <= rand) + 1;      % choose a kernel (1 to #Moves)
    switch kernum % each kernel achieves detailed balance
        case 1 % birth/death
            [Xp,lh] = Simh1(X);
        case 2 % flip label
            [Xp,lh] = Simh2(X);
        case 3 % move point
            [Xp,lh] = Simh3(X);
        case 4 % swap two marked points
            [Xp,lh] = Simh4(X);
    end
    llnew = loglike(Xp);
    lpnew = logprior(Xp);
    lalpha = llnew + lpnew - llold - lpold + lh;
        
    if (rand < exp(lalpha))
        % accepted
        X = Xp;
        llold = llnew; lpold = lpnew;
    end
    % accumulate sattistics here 

end


%-------------------------------------------------------------------------
function ll = loglike(X)
% ll = loglike(X)
% evaluate the log likelihood of a state (up to an additive constant independent of state)

end
%-------------------------------------------------------------------------
function lp = logprior(X)
% lp = logprior(X)
% evaluate the log prior of a state

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh1(X) 
% birth/death  --  return proposed state, and log of the Hastings ratio

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh2(X)
% flip label -- i.e., pick a label uniformly at random, and flip it

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh3(X)
% move point --  pick a point at random, and move its center in a random window

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh4(X)
% swap two marked points (so change positions in front to back ordering)

end
%-------------------------------------------------------------------------
end
