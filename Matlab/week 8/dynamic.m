%%
in = [0:0.01:6];
w = 0.01;
r_in = in/w;
plot(r_in/sum(r_in));
hold on;
r_in = 1./(1 + exp(-in + 5/2));
plot(r_in/sum(r_in));
%%
r = 0;
r = zeros(1,100);
i = [2:1:100];
r(i) = -r(i-1) + 10;
plot(r);
%%
endtime =150;
dt = 0.01;
nt=endtime/dt;
ply=zeros(2,nt+1);
plx=[0:dt:endtime];
r1 = 0.3;
r2 = 0.5;
w = 0.01;
for it=1:nt
    r1 = r1 + 1/(1 + exp(-w*r1 + 5/2));
    ply(1,it) =  r1;
    %r2 = r2 + r2*w;
    %ply(2,it) =  r2;
end
plot(plx,ply);