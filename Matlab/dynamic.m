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
nt=endtime/dt;
ply=zeros(2,nt+1);
plx=[0:dt:endtime];
r1 = 1;
r2 = 1;
w = 0.1;
for it=1:nt
    r1 = r1 + 1/(1 + exp(-w*r1 + 5/2));
    ply(1,it) =  r1;
    %r2 = r2 + r2*w;
    %ply(2,it) =  r2;
end
plot(plx,ply);
%%
dt = 0.01;
endtime =150;
nt=endtime/dt;
ply=zeros(1,nt+1);
plx=[0:dt:endtime];
r = 0;
w = 0.01;
for it=1:nt
    dr = -r + min(6,it*dt*w*6);
    r = r + dr;
    ply(1,it) =  r;
    %r2 = r2 + r2*w;
    %ply(2,it) =  r2;
end
plot(plx,ply);
%%
pic = rand*pi/2;
displayHOG2(pic);
orientations = [0:pi/9:8*pi/9];
results = zeros(1,9);
j = 1;
for i = orientations
    results(:,j) = gaussian(pic,i,6);
    j = j+1;
end
results
dt = 0.01;
endtime =150;
nt=endtime/dt;
ply=zeros(9,nt+1);
plx=[0:dt:endtime];
r = 0;
w = 0.01;
for it=1:nt
    dr = -r + min(6,it*dt*w*6);
    r = r + dr;
    ply(1,it) =  r;
    %r2 = r2 + r2*w;
    %ply(2,it) =  r2;
end
plot(plx,ply);
%%
tau = 6;
r = 0;
a = 1;
drdt = a/tau;
t = [1:1:100];
rs = zeros(1,100);
for s = t
    r = r + drdt;
    drdt = (-r + a)/tau;
    rs(s) = r;
end
plot(t,rs);
    
