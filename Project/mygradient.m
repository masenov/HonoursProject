function [mag,ori] = mygradient(I)
%
% compute image gradient magnitude and orientation at each pixel
%

% compute the derivatives in x and y direction
dx = imfilter(I,[-1,1],'replicate');
dy = imfilter(I,[-1,1]','replicate');

% magnitude of the gradient vector
mag = sqrt(dx.^2+dy.^2);
% orientation of hte gradient vector
ori = atan(dy./dx);
% fix all the NaN values we get from division by zero
ori(isnan(ori))=0;

