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
data = prnist(class_vec, num_objs);

%% Scale and convert to dataset
image_size = [16 16];
preproc = im_box([],0,1)*im_resize([],image_size)*im_box([],1,0);
data_scaled = data*preproc;
img_dataset = prdataset(data_scaled);
prwaitbar off;

%% Selecting all features
          
dataset_with_computed_features = im_features(img_dataset, 'all');
total_num_features = length(dataset_with_computed_features.featlab);

%% Classifier list
parametric_clsf = {fisherc, ldc, qdc, nmc, loglc};
advanced_clsf = {dtc};
non_parametric_clsf = {knnc, parzenc};
num_parametric_clsf = length(parametric_clsf);
num_non_parametric_clsf = length(non_parametric_clsf);
num_advanced_clsf = length(advanced_clsf);

%% Split dataset for feature selection/extraction
% Test for different dataset sizes and pick the largest J criterion
training_dataset_size = 0.5
[trn, tst] = gendat(img_dataset, training_dataset_size);

% Create PCA mapping and generate training and testing set for PCA
[trn_pca, tst_pca] = gendat(img_dataset, training_dataset_size);
featsize = total_num_features;
mapping = scalem([], 'variance')*pcam([], featsize);
w_pca = trn_pca*mapping;
trn_pca_map = trn_pca*w_pca;
tst_pca_map = tst_pca*w_pca;

%% Feature selection

% [w(1),r(1)] = featself(trn, 'NN');
% [w(2),r(2)] = featselb(trn, 'NN');
% [w(3),r(3)] = featsellr(trn, 'NN');
% [w(4),r(4)] = featselp(trn, 'NN');
% [w(5),r(5)] = featselo(trn, 'NN');
% Hangs on my machine!!!

%% Evaluate Feature selection strategies

% Use feature selection techniques to select the right features
featnum = 1:total_num_features;

% for i=1:num_parametric_clsf
%   for j=1:5
%     e_parametric_clsf(i,j) = clevalf(trn*w(j), parametric_clsf(i), featnum, size(trn,1), 1, tst*w(j));
%   end
% end
% 
% for i=1:num_non_parametric_clsf
%   for j=1:5
%     e_non_parametric_clsf(i,j) = clevalf(trn*w(j), non_parametric_clsf(i), featnum, size(trn,1), 1, tst*w(j));
%   end  
% end
% 
% for i=1:num_advanced_clsf
%   for j=1:5
%     e_advanced_clsf(i,j) = clevalf(trn*w(j), advanced_clsf(i), featnum, size(trn,1), 1, tst*w(j));
%   end  
% end
% Hangs on my machine!!!

for i=1:num_parametric_clsf
  e_parametric_clsf(i) = clevalf(trn, parametric_clsf(i), featnum, size(trn,1), 1, tst);
  e_parametric_clsf_pca(i) = clevalf(trn_pca_map, parametric_clsf(i), featnum, size(trn,1), 1, tst_pca_map);
end

for i=1:num_non_parametric_clsf
  e_non_parametric_clsf(i) = clevalf(trn, non_parametric_clsf(i), featnum, size(trn,1), 1, tst);
  e_non_parametric_clsf_pca(i) = clevalf(trn_pca_map, non_parametric_clsf(i), featnum, size(trn,1), 1, tst_pca_map);
end

for i=1:num_advanced_clsf
  e_advanced_clsf(i) = clevalf(trn, advanced_clsf(i), featnum, size(trn,1), 1, tst); 
  e_advanced_clsf_pca(i) = clevalf(trn_pca_map, advanced_clsf(i), featnum, size(trn,1), 1, tst_pca_map); 
end


%% Training classifiers


trn_parametric_clsf = trn*parametric_clsf;
% trn_non_parametric_clsf = trn*non_parametric_clsf;
% trn_advanced_clsf = trn*advanced_clsf;

%% Test classifiers

tested_parametric_clsf = tst*trn_parametric_clsf*testc([]);
% tested_nonparametric_clsf = tst*trn_non_parametric_clsf*testc([]);
% tested_advanced_clsf = tst*trn_advanced_clsf*testc([]);

