% IN4085 - Pattern Recognition
% Winter 2013 - Quarter 2
% Abhishek Sen
% Rodolfo Solera
% Michiel Gerlach

%parameters you can change: num_objs, image_sixe, add im_gauss,
%repset size, number of dimensions

%% Clear existing
addstuff;
close all;
clear all;
clc;
prwaitbar off

%% constants
%Training classifiers
parametric_clsf = {fisherc, ldc, qdc, nmc, loglc};
non_parametric_clsf = {knnc, parzenc};
advanced_clsf = {svc, bpxnc};%, neurc, rnnc, dtc};
dimension = 10;

%% Load datafile
class_vec = 0:9;
%num_objs = [1:5:1000];
num_objs = randi(1000,1,200);
data = prnist(class_vec, num_objs);

%% Convert to dataset
image_size = [32 32];
preproc = im_box([],0,1)*im_resize([],image_size)*im_box([],1,0); % *im_gauss for blurring
data_scaled = data*preproc;
img_dataset = prdataset(data_scaled);
[trn,tst] = gendat(img_dataset,0.5); % 50-50 split in train and testset

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create dissimilarity mapping: _d %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
repset = gendat(trn,dimension/100);          % random 10% for representation !!using too many data cause errors!!
dismap_rep = proxm(repset,'d',1);  % map to dis space by eucl. dist.
trn_d = trn*dismap_rep;            % map the trainset into dis space
tst_d = tst*dismap_rep;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Embedding space - mds_cs: _mds%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dismap_trn = proxm(trn,'d',1);         % define dissimilarity mapping to training set
emmap_mds = trn*dismap_trn*mds_cs([],dimension);   % map training and compute 50D embedding mapping
trn_mds = trn*dismap_trn*emmap_mds;
tst_mds = tst*dismap_trn*emmap_mds;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% embedding space - psem%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dismap_data = proxm(img_dataset,'d',1);
img_dataset_dis = img_dataset*dismap_data;
emmap_psem = psem(img_dataset_dis,dimension,0);
w_ems = img_dataset_dis*emmap_psem;
[trn_psem,tst_psem] = gendat(w_ems,0.5); % 50-50 split in train and testset
% 
% %%%%%%%%%%%%
% %% PCA%%%%%%
% %%%%%%%%%%%%
% pcamap = scalem([],'variance')*pcam([], dimension);
% [trn_p,tst_p] = gendat(img_dataset_dis,0.5);
% pca_trn = trn_p*pcamap;
% trn_p = trn_p*pca_trn;
% tst_p = tst_p*pca_trn;

%%%%%%%%%%%%
%% PCA%%%%%%
%%%%%%%%%%%%
pcadismap = proxm([],'d',1)*scalem([],'variance')*pcam([], dimension);
pca_trn = trn*pcadismap;
trn_p = trn*pca_trn;
tst_p = tst*pca_trn;

%% train classifiers
trn_parametric_clsf_d = trn_d*parametric_clsf;
trn_non_parametric_clsf_d = trn_d*non_parametric_clsf;
%trn_advanced_clsf_d = trn_d*advanced_clsf; 

trn_parametric_clsf_mds = trn_mds*parametric_clsf;
trn_non_parametric_clsf_mds = trn_mds*non_parametric_clsf;
%trn_advanced_clsf_mds = trn_mds*advanced_clsf;   

trn_parametric_clsf_psem = trn_psem*parametric_clsf;
trn_non_parametric_clsf_psem = trn_psem*non_parametric_clsf;
%trn_advanced_clsf_psem = trn_psem*advanced_clsf;     

trn_parametric_clsf_p = trn_p*parametric_clsf;
trn_non_parametric_clsf_p = trn_p*non_parametric_clsf;
%trn_advanced_clsf_p = trn_p*advanced_clsf;     


%% Test classifiers
tested_parametric_clsf_d = tst_d*trn_parametric_clsf_d*testc;
tested_nonparametric_clsf_d = tst_d*trn_non_parametric_clsf_d*testc;
%tested_advanced_clsf_d = tst_d*trn_advanced_clsf_d*testc;

tested_parametric_clsf_mds = tst_mds*trn_parametric_clsf_mds*testc;
tested_nonparametric_clsf_mds = tst_mds*trn_non_parametric_clsf_mds*testc;
%tested_advanced_clsf_mds = tst_mds*trn_advanced_clsf_mds*testc;

tested_parametric_clsf_psem = tst_psem*trn_parametric_clsf_psem*testc;
tested_nonparametric_clsf_psem = tst_psem*trn_non_parametric_clsf_psem*testc;
%tested_advanced_clsf_psem = tst_psem*trn_advanced_clsf_psem*testc;

tested_parametric_clsf_p = tst_p*trn_parametric_clsf_p*testc;
tested_nonparametric_clsf_p = tst_p*trn_non_parametric_clsf_p*testc;
%tested_advanced_clsf_p = tst_p*trn_advanced_clsf_p*testc;

% %% feat sel
% trn_f = trn*dismap_trn;
% tst_f = tst*dismap_trn;
% [w1,r1] = featseli(trn_f,'eucl-m',mf);
% e1 = clevalf(trn_f*w1,clsf,featnum,size(trn_f,1),1,tst_f*w1);
% [w2,r2] = featself(trn_f,'eucl-m',mf);
% e2 = clevalf(trn_f*w2,clsf,featnum,size(trn_f,1),1,tst_f*w2);
% [w3,r3] = featselb(trn_f,'eucl-m');
% e3 = clevalf(trn_f*w3,clsf,featnum,size(trn_f,1),1,tst_f*w3);
     