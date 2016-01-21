function output = vonMises(A,k, actual, pref )
% Function with nice properties to model spiking of orientation selective
% neurons
actual_r = squeeze(repmat(actual,[ones(1,ndims(actual)),size(pref,2)]));
if ndims(actual_r)==3
    pref_r = permute(squeeze(repmat(squeeze(pref),[size(actual_r,1),1,size(actual_r,2)])),[1 3 2]);
elseif ndims(actual_r)==2
    pref_r = reshape(squeeze(repmat(squeeze(pref),[size(actual_r,1),1])),size(actual_r));
end
difference = abs(pref_r-actual_r);
pi_m = pi*ones(size(pref_r));
difference = (difference>pi_m/2).*(pi_m - difference) + (difference<=pi_m/2).*difference;
output = A*exp(k*cos(difference));
plot(squeeze(output(1,2,:)));
xlabel(strcat('Neuron prefered orientation - (x-1)*\pi/',num2str(size(pref,2))));
ylabel('Spiking rate');
title(strcat('Actual orientation  ',num2str(actual(1,2)/pi),'\pi'));
end

