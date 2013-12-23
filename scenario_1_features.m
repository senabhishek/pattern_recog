% IN4085 - Pattern Recognition
% Winter 2013 - Quarter 2
% Abhishek Sen
% Rodolfo Solera
% Michiel Gerlach

%% Clear existing
clc; clear all; close all;

%% Load datafile
class_vec = [0:9];
num_objs = [1:5:1000];
data = prnist(class_vec, num_objs);

%% Scale and convert to dataset
image_size = [16 16];
preproc = im_box([],0,1)*im_gauss*im_resize([],image_size)*im_box([],1,0);
data_scaled = data*preproc;
img_dataset = prdataset(data_scaled);
prwaitbar off;

%% Selecting all features
% Create dataset with all features
dataset_with_computed_features = im_features(img_dataset, 'all');
featsize = length(dataset_with_computed_features.featlab);

%% Classifier list
parametric_clsf = {fisherc, ldc, qdc, nmc, loglc, nmsc, quadrc, pcldc};
advanced_clsf = {dtc};
non_parametric_clsf = {knnc, parzenc, parzendc};

num_parametric_clsf = length(parametric_clsf);
num_non_parametric_clsf = length(non_parametric_clsf);
num_advanced_clsf = length(advanced_clsf);

%% Split dataset for feature selection/extraction
% Test for different dataset sizes and pick the largest J criterion
training_dataset_size = 0.5;
[trn, tst] = gendat(dataset_with_computed_features, training_dataset_size);
[trn_pca, tst_pca] = gendat(dataset_with_computed_features, training_dataset_size);

%% Create feature selection/extraction mappings
% Create PCA mapping and generate training and testing set for PCA
mapping_pca = scalem([], 'variance')*pcam([], featsize);
w_pca = trn_pca*mapping_pca;
trn_pca_map = trn_pca*w_pca;
tst_pca_map = tst_pca*w_pca;

%% Feature selection mapping and mapping train/mapping functions

feature_criterions = {'NN', 'eucl-m', 'maha-m'};
crit = char(feature_criterions(1));
[wf_nn, rf_nn] = featself(trn, crit, featsize);
[wb_nn, rb_nn] = featselb(trn, crit, featsize);
[wlr_nn, rlr_nn] = featsellr(trn, crit, featsize);
[wp_nn, rp_nn] = featselp(trn, crit, featsize);
trn_feat_f_nn = trn * wf_nn;
tst_feat_f_nn = tst * wf_nn;
trn_feat_b_nn = trn * wb_nn;
tst_feat_b_nn = tst * wb_nn;
trn_feat_lr_nn = trn * wlr_nn;
tst_feat_lr_nn = tst * wlr_nn;
trn_feat_p_nn = trn * wp_nn;
tst_feat_p_nn = tst * wp_nn;

crit = char(feature_criterions(2));
[wf_eucl_m, rf_eucl_m] = featself(trn, crit, featsize);
[wb_eucl_m, rb_eucl_m] = featselb(trn, crit, featsize);
[wlr_eucl_m, rlr_eucl_m] = featsellr(trn, crit, featsize);
[wp_eucl_m, rp_eucl_m] = featselp(trn, crit, featsize);
trn_feat_f_eucl_m = trn * wf_eucl_m;
tst_feat_f_eucl_m = tst * wf_eucl_m;
trn_feat_b_eucl_m = trn * wb_eucl_m;
tst_feat_b_eucl_m = tst * wb_eucl_m;
trn_feat_lr_eucl_m = trn * wlr_eucl_m;
tst_feat_lr_eucl_m = tst * wlr_eucl_m;
trn_feat_p_eucl_m = trn * wp_eucl_m;
tst_feat_p_eucl_m = tst * wp_eucl_m;

crit = char(feature_criterions(3));
[wf_maha_m, rf_maha_m] = featself(trn, crit, featsize);
[wb_maha_m, rb_maha_m] = featselb(trn, crit, featsize);
[wlr_maha_m, rlr_maha_m] = featsellr(trn, crit, featsize);
[wp_maha_m, rp_maha_m] = featselp(trn, crit, featsize);
trn_feat_f_maha_m = trn * wf_maha_m;
tst_feat_f_maha_m = tst * wf_maha_m;
trn_feat_b_maha_m = trn * wb_maha_m;
tst_feat_b_maha_m = tst * wb_maha_m;
trn_feat_lr_maha_m = trn * wlr_maha_m;
tst_feat_lr_maha_m = tst * wlr_maha_m;
trn_feat_p_maha_m = trn * wp_maha_m;
tst_feat_p_maha_m = tst * wp_maha_m;

prwaitbar off

%% Feature selection classification analysis for each criterion, and for all classification Techniques
featnum = 1:featsize;

for i=num_parametric_clsf:-1:1
  e_parametric_clsf_f_nn(i) = clevalf(trn_feat_f_nn, parametric_clsf(i), featnum, size(trn_feat_f_nn,1), 1, tst_feat_f_nn);
  e_parametric_clsf_b_nn(i) = clevalf(trn_feat_b_nn, parametric_clsf(i), featnum, size(trn_feat_b_nn,1), 1, tst_feat_b_nn);
  e_parametric_clsf_lr_nn(i) = clevalf(trn_feat_lr_nn, parametric_clsf(i), featnum, size(trn_feat_lr_nn,1), 1, tst_feat_lr_nn);
  e_parametric_clsf_p_nn(i) = clevalf(trn_feat_p_nn, parametric_clsf(i), featnum, size(trn_feat_p_nn,1), 1, tst_feat_p_nn);        
  
  e_parametric_clsf_f_eucl_m(i) = clevalf(trn_feat_f_eucl_m, parametric_clsf(i), featnum, size(trn_feat_f_eucl_m,1), 1, tst_feat_f_eucl_m);
  e_parametric_clsf_b_eucl_m(i) = clevalf(trn_feat_b_eucl_m, parametric_clsf(i), featnum, size(trn_feat_b_eucl_m,1), 1, tst_feat_b_eucl_m);
  e_parametric_clsf_lr_eucl_m(i) = clevalf(trn_feat_lr_eucl_m, parametric_clsf(i), featnum, size(trn_feat_lr_eucl_m,1), 1, tst_feat_lr_eucl_m);
  e_parametric_clsf_p_eucl_m(i) = clevalf(trn_feat_p_eucl_m, parametric_clsf(i), featnum, size(trn_feat_p_eucl_m,1), 1, tst_feat_p_eucl_m);        
  
  e_parametric_clsf_f_maha_m(i) = clevalf(trn_feat_f_maha_m, parametric_clsf(i), featnum, size(trn_feat_f_maha_m,1), 1, tst_feat_f_maha_m);
  e_parametric_clsf_b_maha_m(i) = clevalf(trn_feat_b_maha_m, parametric_clsf(i), featnum, size(trn_feat_b_maha_m,1), 1, tst_feat_b_maha_m);
  e_parametric_clsf_lr_maha_m(i) = clevalf(trn_feat_lr_maha_m, parametric_clsf(i), featnum, size(trn_feat_lr_maha_m,1), 1, tst_feat_lr_maha_m);
  e_parametric_clsf_p_maha_m(i) = clevalf(trn_feat_p_maha_m, parametric_clsf(i), featnum, size(trn_feat_p_maha_m,1), 1, tst_feat_p_maha_m);        
end

for i=num_non_parametric_clsf:-1:1
  e_non_parametric_clsf_f_nn(i) = clevalf(trn_feat_f_nn, non_parametric_clsf(i), featnum, size(trn_feat_f_nn,1), 1, tst_feat_f_nn);
  e_non_parametric_clsf_b_nn(i) = clevalf(trn_feat_b_nn, non_parametric_clsf(i), featnum, size(trn_feat_b_nn,1), 1, tst_feat_b_nn);
  e_non_parametric_clsf_lr_nn(i) = clevalf(trn_feat_lr_nn, non_parametric_clsf(i), featnum, size(trn_feat_lr_nn,1), 1, tst_feat_lr_nn);
  e_non_parametric_clsf_p_nn(i) = clevalf(trn_feat_p_nn, non_parametric_clsf(i), featnum, size(trn_feat_p_nn,1), 1, tst_feat_p_nn);        
  
  e_non_parametric_clsf_f_eucl_m(i) = clevalf(trn_feat_f_eucl_m, non_parametric_clsf(i), featnum, size(trn_feat_f_eucl_m,1), 1, tst_feat_f_eucl_m);
  e_non_parametric_clsf_b_eucl_m(i) = clevalf(trn_feat_b_eucl_m, non_parametric_clsf(i), featnum, size(trn_feat_b_eucl_m,1), 1, tst_feat_b_eucl_m);
  e_non_parametric_clsf_lr_eucl_m(i) = clevalf(trn_feat_lr_eucl_m, non_parametric_clsf(i), featnum, size(trn_feat_lr_eucl_m,1), 1, tst_feat_lr_eucl_m);
  e_non_parametric_clsf_p_eucl_m(i) = clevalf(trn_feat_p_eucl_m, non_parametric_clsf(i), featnum, size(trn_feat_p_eucl_m,1), 1, tst_feat_p_eucl_m);        
  
  e_non_parametric_clsf_f_maha_m(i) = clevalf(trn_feat_f_maha_m, non_parametric_clsf(i), featnum, size(trn_feat_f_maha_m,1), 1, tst_feat_f_maha_m);
  e_non_parametric_clsf_b_maha_m(i) = clevalf(trn_feat_b_maha_m, non_parametric_clsf(i), featnum, size(trn_feat_b_maha_m,1), 1, tst_feat_b_maha_m);
  e_non_parametric_clsf_lr_maha_m(i) = clevalf(trn_feat_lr_maha_m, non_parametric_clsf(i), featnum, size(trn_feat_lr_maha_m,1), 1, tst_feat_lr_maha_m);
  e_non_parametric_clsf_p_maha_m(i) = clevalf(trn_feat_p_maha_m, non_parametric_clsf(i), featnum, size(trn_feat_p_maha_m,1), 1, tst_feat_p_maha_m);        
end

for i=num_advanced_clsf:-1:1
  e_advanced_clsf_f_nn(i) = clevalf(trn_feat_f_nn, advanced_clsf(i), featnum, size(trn_feat_f_nn,1), 1, tst_feat_f_nn);
  e_advanced_clsf_b_nn(i) = clevalf(trn_feat_b_nn, advanced_clsf(i), featnum, size(trn_feat_b_nn,1), 1, tst_feat_b_nn);
  e_advanced_clsf_lr_nn(i) = clevalf(trn_feat_lr_nn, advanced_clsf(i), featnum, size(trn_feat_lr_nn,1), 1, tst_feat_lr_nn);
  e_advanced_clsf_p_nn(i) = clevalf(trn_feat_p_nn, advanced_clsf(i), featnum, size(trn_feat_p_nn,1), 1, tst_feat_p_nn);        
  
  e_advanced_clsf_f_eucl_m(i) = clevalf(trn_feat_f_eucl_m, advanced_clsf(i), featnum, size(trn_feat_f_eucl_m,1), 1, tst_feat_f_eucl_m);
  e_advanced_clsf_b_eucl_m(i) = clevalf(trn_feat_b_eucl_m, advanced_clsf(i), featnum, size(trn_feat_b_eucl_m,1), 1, tst_feat_b_eucl_m);
  e_advanced_clsf_lr_eucl_m(i) = clevalf(trn_feat_lr_eucl_m, advanced_clsf(i), featnum, size(trn_feat_lr_eucl_m,1), 1, tst_feat_lr_eucl_m);
  e_advanced_clsf_p_eucl_m(i) = clevalf(trn_feat_p_eucl_m, advanced_clsf(i), featnum, size(trn_feat_p_eucl_m,1), 1, tst_feat_p_eucl_m);        
  
  e_advanced_clsf_f_maha_m(i) = clevalf(trn_feat_f_maha_m, advanced_clsf(i), featnum, size(trn_feat_f_maha_m,1), 1, tst_feat_f_maha_m);
  e_advanced_clsf_b_maha_m(i) = clevalf(trn_feat_b_maha_m, advanced_clsf(i), featnum, size(trn_feat_b_maha_m,1), 1, tst_feat_b_maha_m);
  e_advanced_clsf_lr_maha_m(i) = clevalf(trn_feat_lr_maha_m, advanced_clsf(i), featnum, size(trn_feat_lr_maha_m,1), 1, tst_feat_lr_maha_m);
  e_advanced_clsf_p_maha_m(i) = clevalf(trn_feat_p_maha_m, advanced_clsf(i), featnum, size(trn_feat_p_maha_m,1), 1, tst_feat_p_maha_m);  
end

%% Feature extraction (PCA) classification analysis

for i=num_parametric_clsf:-1:1
  e_parametric_clsf_pca(i) = clevalf(trn_pca_map, parametric_clsf(i), featnum, size(trn,1), 1, tst_pca_map);
end

for i=num_non_parametric_clsf:-1:1
  e_non_parametric_clsf_pca(i) = clevalf(trn_pca_map, non_parametric_clsf(i), featnum, size(trn,1), 1, tst_pca_map);
end

for i=num_advanced_clsf:-1:1
  e_advanced_clsf_pca(i) = clevalf(trn_pca_map, advanced_clsf(i), featnum, size(trn,1), 1, tst_pca_map); 
end

%% Find minimum classification error for all classifiers
e_parametric = [e_parametric_clsf_f_nn e_parametric_clsf_b_nn e_parametric_clsf_lr_nn ...
                e_parametric_clsf_f_eucl_m e_parametric_clsf_b_eucl_m e_parametric_clsf_lr_eucl_m e_parametric_clsf_p_eucl_m ...
                e_parametric_clsf_f_maha_m e_parametric_clsf_b_maha_m e_parametric_clsf_lr_maha_m e_parametric_clsf_p_maha_m];

e_non_parametric = [e_non_parametric_clsf_f_nn e_non_parametric_clsf_b_nn e_non_parametric_clsf_lr_nn ...
                    e_non_parametric_clsf_f_eucl_m e_non_parametric_clsf_b_eucl_m e_non_parametric_clsf_lr_eucl_m e_non_parametric_clsf_p_eucl_m ...
                    e_non_parametric_clsf_f_maha_m e_non_parametric_clsf_b_maha_m e_non_parametric_clsf_lr_maha_m e_non_parametric_clsf_p_maha_m];

e_advanced = [e_advanced_clsf_f_nn e_advanced_clsf_b_nn e_advanced_clsf_lr_nn ...
              e_advanced_clsf_f_eucl_m e_advanced_clsf_b_eucl_m e_advanced_clsf_lr_eucl_m e_advanced_clsf_p_eucl_m ...
              e_advanced_clsf_f_maha_m e_advanced_clsf_b_maha_m e_advanced_clsf_lr_maha_m e_advanced_clsf_p_maha_m];

e_crit_nn = [e_parametric_clsf_p_nn e_non_parametric_clsf_p_nn e_advanced_clsf_p_nn];

[e_crit_nn_min, e_crit_nn_min_featsize, e_crit_nn_min_index] = find_minimum_error(e_crit_nn);
[e_parametric_pca_min, e_parametric_pca_min_featsize, e_parametric_pca_index] = find_minimum_error(e_parametric_clsf_pca);
[e_non_parametric_pca_min, e_non_parametric_pca_min_featsize, e_non_parametric_pca_index] = find_minimum_error(e_non_parametric_clsf_pca);
[e_advanced_pca_min, e_advanced_pca_min_featsize, e_advanced_pca_index] = find_minimum_error(e_advanced_clsf_pca);
[e_parametric_min, e_parametric_min_featsize, e_parametric_index] = find_minimum_error(e_parametric);
[e_non_parametric_min, e_non_parametric_min_featsize, e_non_parametric_index] = find_minimum_error(e_non_parametric);
[e_advanced_min, e_advanced_featsize, e_advanced_index] = find_minimum_error(e_advanced);

e_min = [e_parametric_clsf_pca(e_parametric_pca_index) ...
          e_non_parametric_clsf_pca(e_non_parametric_pca_index) ...
          e_advanced_clsf_pca(e_advanced_pca_index) ...
          e_parametric(e_parametric_index) ...
          e_non_parametric(e_non_parametric_index) ...
          e_advanced(e_advanced_index)];

[e_overall_min, e_overall_min_featsize, e_overall_min_index] = find_minimum_error(e_min);

if (e_crit_nn_min < e_overall_min)
  overall_min_achieved_with_crit_nn = true;
  e_overall_min = e_crit_nn_min;
  e_overall_min_featsize = e_crit_nn_min_featsize;
  e_overall_min_index = e_crit_nn_min_index;
end

%% Plot error curves (PCA)
% figure(1);
% parametric_clsf_pca_e_plot = plote(e_parametric_clsf_pca);
% figure(2);
% non_parametric_clsf_pca_e_plot = plote(e_non_parametric_clsf_pca);
% figure(3);
% advanced_clsf_pca_e_plot = plote(e_advanced_clsf_pca);

%% Plot error curves (Feature selection + classifier)

% % Parametric
% close all;
% 
% % NN
% figure(4);
% plote(e_parametric_clsf_f_nn);
% figure(5);
% plote(e_parametric_clsf_b_nn);
% figure(6);
% plote(e_parametric_clsf_lr_nn);
% figure(7);
% plote(e_parametric_clsf_p_nn);
% 
% % Eucl-m
% figure(8);
% plote(e_parametric_clsf_f_eucl_m);
% figure(9);
% plote(e_parametric_clsf_b_eucl_m);
% figure(10);
% plote(e_parametric_clsf_lr_eucl_m);
% figure(11);
% plote(e_parametric_clsf_p_eucl_m);
% 
% % Maha-m
% figure(8);
% plote(e_parametric_clsf_f_maha_m);
% figure(9);
% plote(e_parametric_clsf_b_maha_m);
% figure(10);
% plote(e_parametric_clsf_lr_maha_m);
% figure(11);
% plote(e_parametric_clsf_p_maha_m);
% 
% %% Non-Parametric
% close all;
% 
% % NN
% figure(12);
% plote(e_non_parametric_clsf_f_nn);
% figure(13);
% plote(e_non_parametric_clsf_b_nn);
% figure(14);
% plote(e_non_parametric_clsf_lr_nn);
% figure(15);
% plote(e_non_parametric_clsf_p_nn);
% 
% % Eucl-m
% figure(16);
% plote(e_non_parametric_clsf_f_eucl_m);
% figure(17);
% plote(e_non_parametric_clsf_b_eucl_m);
% figure(18);
% plote(e_non_parametric_clsf_lr_eucl_m);
% figure(19);
% plote(e_non_parametric_clsf_p_eucl_m);
% 
% % Maha-m
% figure(20);
% plote(e_non_parametric_clsf_f_maha_m);
% figure(21);
% plote(e_non_parametric_clsf_b_maha_m);
% figure(22);
% plote(e_non_parametric_clsf_lr_maha_m);
% figure(23);
% plote(e_non_parametric_clsf_p_maha_m);
% 
% %% Advanced
% close all;
% 
% % NN
% figure(24);
% plote(e_advanced_clsf_f_nn);
% figure(25);
% plote(e_advanced_clsf_b_nn);
% figure(26);
% plote(e_advanced_clsf_lr_nn);
% figure(27);
% plote(e_advanced_clsf_p_nn);
% 
% % Eucl-m
% figure(28);
% plote(e_advanced_clsf_f_eucl_m);
% figure(29);
% plote(e_advanced_clsf_b_eucl_m);
% figure(30);
% plote(e_advanced_clsf_lr_eucl_m);
% figure(31);
% plote(e_advanced_clsf_p_eucl_m);
% 
% % Maha-m
% figure(32);
% plote(e_advanced_clsf_f_maha_m);
% figure(33);
% plote(e_advanced_clsf_b_maha_m);
% figure(34);
% plote(e_advanced_clsf_lr_maha_m);
% figure(35);
% plote(e_advanced_clsf_p_maha_m);

