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
          
working_featset = {'Area','Centroid'};         
dataset_with_computed_features = im_features(img_dataset, 'all');

%% Training classifiers
parametric_clsf = {fisherc, ldc, qdc, nmc, loglc};
non_parametric_clsf = {knnc, parzenc};
advanced_clsf = {svc, bpxnc};%, neurc, rnnc, dtc};

[trn,tst] = gendat(dataset_with_computed_features, 0.5);
trn_parametric_clsf = trn*parametric_clsf;
trn_non_parametric_clsf = trn*non_parametric_clsf;
trn_advanced_clsf = trn*advanced_clsf;

%% Test classifiers

tested_parametric_clsf = tst*trn_parametric_clsf*testc([]);
tested_nonparametric_clsf = tst*trn_non_parametric_clsf*testc([]);
tested_advanced_clsf = tst*trn_advanced_clsf*testc([]);

