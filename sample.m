% IN4085 - Pattern Recognition
% Winter 2013 - Quarter 2
% Abhishek Sen
% Rodolfo Solera
% Michiel Gerlach

% Clear existing
clc; clear all; close all;

% Load datafile
class_vec = [0:9];
num_objs = [1:10];
data = prnist(class_vec, num_objs);

% Convert to dataset
image_size = [128 128];
preproc = im_box([],0,1)*im_norm*im_resize([],image_size)*im_box([],1,0);
data_scaled = data*preproc;
img_dataset = prdataset(data_scaled);
show(img_dataset);
prwaitbar off;
%scatterd(img_dataset);


