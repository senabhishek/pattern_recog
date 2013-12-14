% IN4085 - Pattern Recognition
% Winter 2013 - Quarter 2
% Abhishek Sen
% Rodolfo Solera
% Michiel Gerlach

%% Clear existing
clc; clear all;

%% Load datafile
class_vec = [0:9];
num_objs = [1:5:1000];
%num_objs = [1:100:1000];
data = prnist(class_vec, num_objs);

%% Scale and convert to dataset
image_size = [16 16];
preproc = im_box([],0,1)*im_resize([],image_size)*im_box([],1,0);
data_scaled = data*preproc;
img_dataset = prdataset(data_scaled);
prwaitbar off;

%% Feature selection
master_featset = {'Area','EulerNumber','Orientation','BoundingBox','Extent','Perimeter','Centroid',...
            'Extrema','PixelIdxList','ConvexArea','FilledArea','PixelList','ConvexHull',...
            'FilledImage','Solidity','ConvexImage','SubarrayIdx','Eccentricity',...
            'MajorAxisLength','EquivDiameter','MinorAxisLength'};
          
working_featset = {'Area','Centroid'};         
dataset_with_computed_features = im_features(img_dataset, working_featset);
classifier = [fisherc qdc];

