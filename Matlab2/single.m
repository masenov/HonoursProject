close all;
receptive_field = pi;
displayHOG2(receptive_field);
orientations = [0:pi/9/20:8*pi/9];
results = zeros(1,9);
results_von = zeros(1,9);
x = [-2*pi/9:0.01:pi+2*pi/9];
j = 1;
for i = orientations
    results1 = gaussian(receptive_field,i,6);
    results2 = gaussian(receptive_field+pi,i,6);
    results3 = gaussian(receptive_field-pi,i,6);
    results(j) = max([results1,results2,results3]);
    results_von(j) = exp(8*cos(receptive_field-i))
    j = j+1;
end
figure;
plot([results_von results_von(1)]);