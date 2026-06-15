% script file to make fake image of cells

npix = 100;              % make npix*npix image
slide = ones(npix,npix); % 1 = bright

ncellmax = 10;           % max nr cells
cellrad = 9.5;           % radius of cells
stddev = 0.1;            % noise standard deviation


ncell = 5+ ceil((ncellmax-5)*rand);   % random number of cells (at least 5)
pbad = 0.25+ 0.5*rand;                % prob cell is bad

xycell = ceil(npix*rand(2,ncell));
for icell = 1:ncell;
    if rand < pbad
        slide = putbad(slide,xycell(1,icell),xycell(2,icell),cellrad);
    else
        slide = putgood(slide,xycell(1,icell),xycell(2,icell),cellrad);
    end
end

slide = slide + stddev*randn(size(slide));
colormap(gray(256))
image(slide,'CDataMapping','scaled')
% save slide slide