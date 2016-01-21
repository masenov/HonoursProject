for index=1:1:timesteps
    displayHOG2(direction(:,:,index));
    h = getframe;
    imwrite(h.cdata, strcat('frames/',num2str(index),'.png'));
    close all;
end