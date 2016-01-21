function matrix = covarianceMatrix( rate_matrix, distance_factor, orientation_factor )
%covarianceMatrix Generate covariance matrix for the weights





vector_length = numel(rate_matrix);
matrix = zeros(vector_length,vector_length);
for i=1:vector_length
    for j=1:vector_length
        if i==j
            continue
        else
            i_coord = calculateCoordinates(i,size(rate_matrix));
            j_coord = calculateCoordinates(j,size(rate_matrix));
            distance = sqrt((i_coord(1)-j_coord(1))^2 + (i_coord(2)-j_coord(2))^2);
            angle_diff = min(mod(i_coord(3)-j_coord(3),size(rate_matrix,3)), mod(j_coord(3)-i_coord(3),size(rate_matrix,3)));
            matrix(i,j) = min(distance_factor,distance_factor*(1/exp(distance))) + min(orientation_factor,orientation_factor*(1/exp(angle_diff)));
            matrix(j,i) = min(distance_factor,distance_factor*(1/exp(distance))) + min(orientation_factor,orientation_factor*(1/exp(angle_diff)));
        end
    end
end
