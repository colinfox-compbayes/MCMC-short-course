function [kvec,acc] = counting_MCMC(nsamp)
% Function file implements MCMC sampling counting assignment
% returning number of 'good' and 'bad' cells
%
% [kvec] = counting_MCMC(nsamp)
%

% Colin Fox, 23 July 2019

% Read in image file
slide = double(imread('slide.tif'))/255; % this is taken as data. Dividing by 255 undoes the scale when writing a tif to UINT8
% plot the slide
figure(1), clf
colormap(gray(256))
image(slide,'CDataMapping','scaled')
axis('square')

MoveRatio = [30 1 1 1]; 
MoveProb = cumsum(MoveRatio(1:end-1)/sum(MoveRatio));

% constants from makefake
npix = 100;              % make npix*npix image
cellrad = 9.5;           % radius of cells
stddev = 0.1;            % noise standard deviation

showeach = 10;           % subsampling for showing state

mean_num = 10;           % my guess at mean number of points
pgeo = 1/(1+mean_num);   % parameter in geometric distribution

X = [];                  % starting state is null state
llold = loglike(X);
lpold = logprior(X);

prop = zeros(size(MoveRatio));       % collect number of proposals
acc = zeros(size(MoveRatio));        % collect acceptance ratios
kvec = zeros(1,nsamp);               % collect marginal state over number
kvec(1) = size(X,2);
        
for iter = 2:nsamp
    kernum = sum(MoveProb <= rand) + 1;      % choose a kernel (1 to #Moves)
    prop(kernum) = prop(kernum) + 1;

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
        acc(kernum) =  acc(kernum) + 1;
        
        X = Xp;
        llold = llnew; lpold = lpnew;
    end
    kvec(iter)= size(X,2);
    if mod(iter,showeach) == 0; % show the current state
      figure(2), colormap(gray(256))
      image(makeimage(X,npix),'CDataMapping','scaled')
      axis('square'), drawnow
    end

end

% calculate acceptance rates
acc = [acc./(prop+eps), sum(acc)/nsamp];


%-------------------------------------------------------------------------
function ll = loglike(X)
% ll = loglike(X)
% evaluate the log likelihood of a state (up to an additive constant independent of state)

% Colin Fox, 23 July 2019

ll = (-1/(2*stddev^2))*sum(sum((makeimage(X,npix) - slide).^2));
% note that writing to tiff clips the original observations to [0,1], hence this is not quite the correct likelihood.
% compare hist(slide(:),100) before and after writing to .tif to see the clipping

end
%-------------------------------------------------------------------------
function lp = logprior(X)
% lp = logprior(X)
% evaluate the log prior of a state

% Colin Fox, 23 July 2019

k = size(X,2); 
if k >= 0
  lp = k*log(1-pgeo) + log(pgeo);
else
  lp = -inf;
end

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh1(X)
% [Xp,lh] = Simh1(X)  % return proposed state, and log of the Hastings ratio
% birth/death

% Colin Fox, 23 July 2019

k = size(X,2); 

if rand < 0.5
  % birth -- pick a random place from 1 to k+1, and insert a valid marked point
  inew = ceil(rand*(k+1));                                % insert new object before index inew, or at end if inew = k+1
  xynew = ceil(npix*rand(2,1)); lnew = ceil(2*rand-1);    % pick a random marked-point from the prior
  Xp = [X(:,1:inew-1),[xynew;lnew],X(:,inew:end)];        % insert new point
else
  % death -- uniformly at random pick a marked point and zap it
  if k == 0, Xp = X; lh = 0; return, end
  izap = ceil(rand*k);                                    % zap object izap
  Xp = [X(:,1:izap-1),X(:,izap+1:end)];                   % insert new point
end

lh = 0; % this proposal is symmetric

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh2(X)
% [Xp,lh] = Simh2(X)
% flip label
% i.e., pick a label uniformly at random, and flip it

k = size(X,2); 
if k == 0, Xp = X; lh = 0; return, end
iflip = ceil(rand*k);                                      % pick an index
Xp = X;
Xp(3,iflip) = 1-X(3,iflip);                                % and flip 1 <--> 0

lh = 0; % this proposal is symetric

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh3(X)
% [Xp,lh] = Simh3(X)
% move point
% pick a point at random, and move its center in a random window

k = size(X,2); 
if k == 0, Xp = X; lh = 0; return, end

w = 6;                                                     % window size in pixels
imove = ceil(rand*k);                                      % pick an index
xymove = round(w*(rand(2,1)-0.5));                         % amount to move
Xp = X;
Xp(1:2,imove) = Xp(1:2,imove)+xymove;                      % and move point

lh = 0; % this proposal is symetric

end
%-------------------------------------------------------------------------
function [Xp,lh] = Simh4(X)
% [Xp,lh] = Simh4(X)
% swap two marked points (so change positions in front to back ordering)

k = size(X,2); 
if k == 0, Xp = X; lh = 0; return, end
i12 = ceil(rand(1,2)*k);                                 % pick two indices
Xp = X;
Xp(:,i12) = X(:,i12([end:-1:1]));                        % and swap marked points 

lh = 0; % this proposal is symetric

end
%-------------------------------------------------------------------------
end
