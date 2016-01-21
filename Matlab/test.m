close all;
size_x = 10;
size_y = 10;
pic = zeros(size_x,size_y);
reshaped_pic = pic(:);
size_image = size(pic,1)*size(pic,2);
for i = 1:(size_image)
    %reshaped_pic(i) = rand*pi;
    pi/2/size_image*i
    reshaped_pic(i) = (pi/size_image)*i;
end
%pic = reshape(reshaped_pic',[size_x size_y])';
displayHOG2(reshaped_pic);
pic = reshaped_pic;
size_x = size_image;
size_y = 1;
orientations = [0:pi/9:8*pi/9];
x = [-2*pi/9:0.01:pi+2*pi/9];
figure;
results = zeros(size_x,size_y,9);
%x = mod(x,pi);
j = 1;
for i = orientations
    results(:,:,j) = gaussian(pic,i,6);
    filters = gaussian(x,i,6)*15;
    mod_x = mod(x,pi);
    plot(mod_x(filters>0.001),filters(filters>0.001)); hold on;
    j = j+1;
end
results2 = bsxfun(@times, squeeze(results), reshape(orientations,[1,9]) );
results2 = sum(results2,2)./sum(squeeze(results),2);
figure;
displayHOG2(results2);
displayHOG2(results2);