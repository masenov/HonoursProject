orientations = [0:pi/9:8*pi/9]
ac_oriet = pi/4;
responses = zeros(9,100);
k = 1;
for i=orientations
  tau = 6;
  r = 0;
  a = vonMises(0.25, 3, ac_oriet,i)
  drdt = a/tau;
  t = [1:1:100];
  rs = zeros(1,100);
  for s = t
      r = r + drdt;
      drdt = (-r + a)/tau;
      rs(s) = r;
  end
  responses(k,:) = rs;
  k = k + 1;
end
plot(t,responses);
figure;
plot(orientations*floor(responses)./sum(floor(responses),1));