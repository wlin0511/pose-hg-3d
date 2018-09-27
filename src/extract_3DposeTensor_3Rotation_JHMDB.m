clear all
close all
addpath('/home/lin7lr/3Dpose/towards_3D_Human/src')


splitNr = 3;

h5_file_test = ['/home/lin7lr/test_pose/h5/JHMDB/train_test_subplit',num2str(splitNr),'_29node_resize/testlist_%dpos.h5'];

dataset= 'JHMDB';  % 'Airport'
smooth_flag = 0;
scaleNorm_flag = 0;

testlist_rgb = ['/home/lin7lr/test_pose/list/JHMDB/train_test_subsplit',num2str(splitNr),'/rgb_testlist_subsplit',num2str(splitNr),'.txt'];
test_sequences = get_sequence_paths(testlist_rgb, dataset, smooth_flag, scaleNorm_flag);


% node = [9 13 12 11 12 13 9 7 3 7 9 7 4 7 9 14 15 16 15 14 9];
% node = [9 13 12 11 12 13 9 7 9 14 15 16 15 14 9];  % 15 nodes

node = [7 3 2 1 2 3 7 4 5 6 5 4 7 9 10 9 13 12 11 12 13 9 14 15 16 15 14 9 7];  % 29 nodes
% node = [9 13 12 11 12 13 9 7 3 2 1 2 3 ];
dim1 = 10;
dim2 = length(node)*3;

rotation_flag = 1;
vel_norm = 1;
acc_norm = 0;

% %% train
% [pose_4d_train, train_labels, ~, sequence_counter_train] = pose_tensor_formation_3Rotation(dim1,dim2, train_sequences, node,  rotation_flag, vel_normalization, acc_normalization,dataset);
% h5_file_train = sprintf(h5_file_train, sequence_counter_train);
% h5create(h5_file_train, '/dataset', size(pose_4d_train))  % create HDF5 data set
% h5write(h5_file_train, '/dataset', pose_4d_train)
% h5create(h5_file_train,'/label',size(train_labels),'Datatype','double');
% h5write(h5_file_train,'/label',train_labels);

%% test
%[pose_4d_test, test_labels, ~, sequence_counter_test] = pose_tensor_formation_3Rotation(dim1,dim2, test_sequences, node, rotation_flag, vel_normalization, acc_normalization,dataset);
[pose_4d_test, test_labels, ~, sequence_counter_test] = pose_tensor_formation_3Rotation_resize(dim1,dim2, test_sequences, node, rotation_flag, vel_norm, acc_norm,dataset);
h5_file_test = sprintf(h5_file_test, sequence_counter_test);
h5create(h5_file_test, '/dataset', size(pose_4d_test))  % create HDF5 data set
h5write(h5_file_test, '/dataset', pose_4d_test)
h5create(h5_file_test,'/label',size(test_labels),'Datatype','double');
h5write(h5_file_test,'/label',test_labels);





