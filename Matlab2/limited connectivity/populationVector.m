function [direction,magnitude] = populationVector( orientations, r, nosn, timesteps )
m = size(r,1);
n = size(r,2);
orientations = 2 * orientations;
sin_o = sin(orientations);
cos_o = cos(orientations);
r = reshape(r,[m*n nosn timesteps]);
r_x = zeros(m*n,timesteps);
r_y = zeros(m*n,timesteps);
for i=[1:1:m*n]
    r_x(i,:) = sin_o * squeeze(r(i,:,:));
    r_y(i,:) = cos_o * squeeze(r(i,:,:));
end

%quiver(zeros(1,nosn+1),zeros(1,nosn+1),[sin_o.*r(1,:,1),r_x(1,1)*magnitude(1,1,1)],[cos_o.*r(1,:,1),r_y(1,1)*magnitude(1,1,1)]);
magnitude = sqrt(r_x.^2 + r_y.^2);
r_x = r_x./magnitude;
r_y = r_y./magnitude;
direction = (asin(r_x)>0).*acos(r_y) + (asin(r_x)<0).*(2*pi-acos(r_y));
direction = real(direction / 2);
direction = reshape(direction,[m n timesteps]);
magnitude = reshape(magnitude,[m n timesteps]);
end

