function Xslide = makeimage(X,npix)
%
% Xslide = makeimage(X,npix)
%
% function file that takes a state X and produces the image (slide) with
% associated good and bad cells.

% Colin Fox, 23 July 2019

% State is 3xk ordered marked points: ( [x1;y1;l1], ... , [xk;yk;lk] ) 
% k is number of points
% each l is 0 = 'bad', 1 = 'good'
% first point is infront, last is at back

cellrad = 9.5;           % radius of cells

Xslide = ones(npix,npix); % make npix*npix image, 1 = bright

k = size(X,2);
if  k == 0; return % nothing to do
else
  for icell = k+1-[1:k] % count backwards
     if X(3,icell) == 0
        Xslide = putbad(Xslide,X(1,icell),X(2,icell),cellrad);
    else
        Xslide = putgood(Xslide,X(1,icell),X(2,icell),cellrad);
    end
  end
end
