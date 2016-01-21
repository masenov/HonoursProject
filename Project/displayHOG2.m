function displayHOG2(hog)
%Visualizing HOG

histogram = zeros(size(hog,1)*16,size(hog,2)*16);
imshow(histogram);
hold on;
lineLength = 8;
%angle = 5*pi/9;


for i=1:size(hog,1)
    for j=1:size(hog,2)
        angle = hog(i,j)*pi;
        p1 = [(i-1)*16+8-lineLength * sin(angle),(j-1)*16+8-lineLength * cos(angle)];
        p2 = [(i-1)*16+8+lineLength * sin(angle),(j-1)*16+8+lineLength * cos(angle)];
        plot([p1(2),p2(2)],[p1(1),p2(1)],'Color',[1 1 1],'LineWidth',1);
    end
end

end

