close all; clear;
m = 2;
n = 3;
tau = 6;
orientations = [0:pi/9:8*pi/9];
ac_orient = pi*rand(m,n);
responses = zeros(9,100);
t = 1:1:100;

as = vonMises(0.25,3,ac_orient,orientations);
as = as(:);
r = zeros(size(as));
drdt = as/tau;
rs = zeros([size(as),length(t)]);
weight_matrix = covarianceMatrix(ones(m,n,),0.005,0.005);
for s = t
    r = r + drdt;
    %pad_r = padarray(r,[1 1 0]);
    %near_r = pad_r(1:m,1:n,:) + pad_r(2:m+1,1:n,:) + pad_r(3:m+2,1:n,:) + pad_r(1:m,2:n+1,:) + pad_r(3:m+2,2:n+1,:) + pad_r(1:m,3:n+2,:) + pad_r(2:m+1,3:n+2,:) + pad_r(3:m+2,3:n+2,:);
    drdt = (-r + as)/tau + r*weight_matrix;
    rs(:,1,s) = r;
end

figure;
r_first_neuron = squeeze(rs(1,1,:,:));
plot(t,r_first_neuron);
title('Orientation selective neurons response rates');
figure;
[direction,magnitude] = populationVector(orientations,rs);
displayHOG2(ac_orient);
figure;
displayHOG2(direction(:,:,100));