close all; clear;
m = 2;
n = 2;
tau = 6;
% Number of orientation selective neurons
nosn = 9;
timesteps = 10;
% for vonMises function
k = 0.25;
A = 3;


orientations = [0:pi/nosn:(nosn-1)*pi/nosn];
%ac_orient = pi*rand(m,n);
ac_orient = (1:4)*pi/4;
ac_orient = reshape(ac_orient, [2 2]);
responses = zeros(nosn,timesteps);
t = 1:1:timesteps;

as = vonMises(k,A,ac_orient,orientations);
r = zeros(size(as));
drdt = as/tau;
rs = zeros([size(as),length(t)]);
for s = t
    r = r + drdt;
    pad_r = padarray(r,[1 1 0]);
    near_r = pad_r(1:m,1:n,:) + pad_r(2:m+1,1:n,:) + pad_r(3:m+2,1:n,:) + pad_r(1:m,2:n+1,:) + pad_r(3:m+2,2:n+1,:) + pad_r(1:m,3:n+2,:) + pad_r(2:m+1,3:n+2,:) + pad_r(3:m+2,3:n+2,:);
    drdt = (-r + as)/tau + 0.02*near_r;
    rs(:,:,:,s) = r;
end

    
figure;
r_first_neuron = squeeze(rs(1,1,:,:));
plot(t,r_first_neuron);
title('Orientation selective neurons response rates');
figure;
[direction,magnitude] = populationVector(orientations,rs, nosn, timesteps);
displayHOG2(ac_orient);
figure;
displayHOG2(direction(:,:,timesteps));