A = 0.25;
k = 3;
x = 0:0.01:pi;
plot(x,A*exp(k*cos(x)))
title('Nice properties of von Mises function for calculating spiking based on differences in angles');
xlabel('Angle difference');
ylabel('Spiking');
%%
quiver(zeros(1,100),zeros(1,100),cos(0.05:pi/50:2*pi),sin(0.05:pi/50:2*pi))
%%
quiver(zeros(1,3),zeros(1,3),[sin(0),sin(2*pi/7),sin(pi/7)+sin(pi/7)],[cos(0),cos(2*pi/7),cos(pi/7)+cos(pi/7)])
