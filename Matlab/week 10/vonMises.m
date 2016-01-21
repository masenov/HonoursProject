function output = vonMises(A,k, pref, actual )
% Function with nice properties to model spiking of orientation selective
% neurons
 output = A*exp(k*cos(abs(pref-actual)));

end

