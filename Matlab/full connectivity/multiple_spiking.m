close all; clear;
m = 10;
n = 10;
tau = 6;
% Number of orientation selective neurons
nosn = 9;
timesteps = 5;
% for vonMises function
k = 0.25;
A = 3;
% for the weight matrix
distance_scaling = 0.00005;
orientation_scaling = 0.00005;

orientations = [0:pi/nosn:(nosn-1)*pi/nosn];
ac_orient = pi*rand(m,n);
responses = zeros(nosn,timesteps);
t = 1:1:timesteps;

as = vonMises(k,A,ac_orient,orientations);
as = as(:);
r = zeros(size(as));
drdt = as/tau;
rs = zeros([size(as),length(t)]);
weight_matrix = covarianceMatrix(ones(m,n,nosn),distance_scaling,orientation_scaling);
for s = t
    r = r + drdt;
    %pad_r = padarray(r,[1 1 0]);
    %near_r = pad_r(1:m,1:n,:) + pad_r(2:m+1,1:n,:) + pad_r(3:m+2,1:n,:) + pad_r(1:m,2:n+1,:) + pad_r(3:m+2,2:n+1,:) + pad_r(1:m,3:n+2,:) + pad_r(2:m+1,3:n+2,:) + pad_r(3:m+2,3:n+2,:);
    drdt = (-r + as)/tau + weight_matrix*r;
    rs(:,1,s) = r;
end
rs = reshape(rs, [m n nosn timesteps]);
as = reshape(as, [m n nosn]);
[direction,magnitude] = populationVector(orientations,rs, nosn, timesteps);
for i=1:1:timesteps
    current_rate = direction(:,:,i);
    displayHOG2(current_rate,i);
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
%% Print weight matrix
set(gca,'fontize',14);
h1 = figure;
imagesc(weight_matrix)
print(h1,'-djpeg','-r500','filename')
h2 = figure;
imagesc(weight_matrix(1:100,1:100))
print(h2,'-djpeg','-r500','filename2')
h3 = figure;
imagesc(weight_matrix(1:10,1:10))
print(h3,'-djpeg','-r500','filename3')
print(h1,'-djpeg','-r500','filename4')