function coord = calculateCoordinates( i, size_matrix )

z_size = fix((i-1)/(size_matrix(1)*size_matrix(2))) + 1;
remainder = mod(i-1,(size_matrix(1)*size_matrix(2)));
y_size = fix(remainder/size_matrix(1)) + 1;
x_size = mod(remainder,size_matrix(1)) + 1;
coord = [x_size,y_size,z_size];
end

