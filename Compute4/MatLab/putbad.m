function a = putbad(a,x,y,r)
% Put a 'bad' cell in image a

[m,n] = size(a);

% This code is not optimized for speed
[mm,nn] = meshgrid(1:m,1:n);

d2 = (mm - x).^2 + (nn - y).^2;
a(find((mm - y).^2 + (nn - x).^2 <= r^2)) = 0.5;
a(find((mm - y).^2 + (nn - x).^2 <= (r/2)^2)) = 1;