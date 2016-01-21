function y = gaussian(x, mean, var)
%Gaussian Calculate gaussian for given mean and variance
y = (1/(sqrt(2*pi)*var))*exp(-(x-mean).^2/2*var.^2);

end

