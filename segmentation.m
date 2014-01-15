close all;
clear all
clc
%% segment image
folderName = 'Rodolfo'; %directory to store images in
rawImage = imread('marina.jpg');  %# Load the image
%%
figure(1); imshow(rawImage);
Img = rgb2gray(rawImage);       %grayscale image
level = graythresh(Img);        %find a segmentation threshold
Img_b = ~im2bw(Img,level);         %convert the image to binary
Img_b = bwareaopen(Img_b,30);  %delete small objects
figure(2);imshow(Img_b);
% Cc = bwconncomp(Img);
Cc = regionprops(Img_b, 'BoundingBox', 'Extrema');
% components = Cc.PixelIdxList;                 %get a structure that contains the linear indices of each identified component in its field

%%%
%%%%%% sort them:  http://blogs.mathworks.com/steve/2008/03/25/bwlabel-search-order/ %%%%%%%%%%
%%%
extrema = cat(1, Cc.Extrema);
left_most_bottom = extrema(6:8:end, :);
left = left_most_bottom(:, 1);
bottom = left_most_bottom(:, 2);
% quantize the bottom coordinate
bottom = 6 * round(bottom / 6);
[sorted, sort_order] = sortrows([bottom left]);
Cc_sort = Cc(sort_order);
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%

hold on;

mkdir(folderName);

for i=0:length(Cc_sort)-1
%     [x y] = ind2sub(size(Img),cell2mat(components(i)));
%     xmin=min(x); xmax=max(x); ymin=min(y); ymax=max(y); w = xmax-xmin+4; h=ymax-ymin+4;
    if mod(i,10) == 0
        subfolderName = sprintf('%s/digit_%d',folderName,round(i/10));
        mkdir(subfolderName);
    end
    rectangle('Position',Cc_sort(i+1).BoundingBox, 'EdgeColor','r'); %just for showing
    Img_part = imcrop(Img_b,Cc_sort(i+1).BoundingBox);
    file_name = sprintf('%s/img%d.bmp',subfolderName,i);
    imwrite(Img_part,file_name,'BMP');
end
hold off;
%% create datafile
own_data = prdatafile(sprintf('./%s',folderName));
%% convert to dataset
image_size = [32 32];
preproc = im_box([],0,1)*im_resize([],image_size)*im_box([],1,0); % *im_gauss for blurring, don't use im_rotate
own_data_scaled = own_data*preproc;
img_own_dataset = prdataset(own_data_scaled);
figure(3); show(img_own_dataset);

% %%
% maxValue = double(max(rawImage(:)));     %# Find the maximum pixel value
% N = 50;                                  %# Threshold number of white pixels
% boxIndex = sum(rawImage) < N*maxValue;   %# Find columns with fewer white pixels
% boxImage = rawImage;                     %# Initialize the box image
% boxImage(:,boxIndex) = 0;                %# Set the indexed columns to 0 (black)
% dilatedIndex = conv(double(boxIndex),ones(1,5),'same') > 0;  %# Dilate the index
% dilatedImage = rawImage;                 %# Initialize the dilated box image
% dilatedImage(:,dilatedIndex) = 0;        %# Set the indexed columns to 0 (black)
% 
% %# Display the results:
% subplot(3,1,1);
% imshow(rawImage);
% title('Raw image');
% subplot(3,1,2);
% imshow(boxImage);
% title('Boxes placed over numbers');
% subplot(3,1,3);
% imshow(dilatedImage);
% title('Dilated boxes placed over numbers');
% 
% %%
% I = imread('rice.png'); 
% figure, imshow(I)
% BW = im2bw(I, graythresh(I)); 
% CC = bwconncomp(BW);
% L = labelmatrix(CC);
% RGB = label2rgb(L);
% figure, imshow(RGB)