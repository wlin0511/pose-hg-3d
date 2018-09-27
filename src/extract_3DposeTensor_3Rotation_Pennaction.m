clear all
close all
addpath('/home/lin7lr/3Dpose/towards_3D_Human/src')


%  h5_file_train = '/home/lin7lr/test_pose/h5/Penn_action/train_test_29node_smoothed_resize/trainlist_%dpos.h5';
% h5_file_val = '/home/lin7lr/test_pose/h5/5734_5735_Activity_3D_allRGB_15nodes_Normalization_3Rotation_2classes/vallist_%dpos.h5';
h5_file_test = '/home/lin7lr/test_pose/h5/Penn_action/train_test_29node_smoothed_resize/testlist_%dpos.h5';
% 

dataset= 'Penn_action';  % 'Airport'
smooth_flag = 1;
scaleNorm_of_all_limbs_flag = 0;
% trainlist_rgb = '/home/lin7lr/test_pose/list/Penn_action/rgb_trainlist.txt';
% train_sequences = get_sequence_paths(trainlist_rgb,dataset,smooth_flag, scaleNorm_flag);

testlist_rgb = '/home/lin7lr/test_pose/list/Penn_action/rgb_testlist.txt';
test_sequences = get_sequence_paths(testlist_rgb, dataset,smooth_flag, scaleNorm_of_all_limbs_flag);


% node = [9 13 12 11 12 13 9 7 3 7 9 7 4 7 9 14 15 16 15 14 9];
% node = [9 13 12 11 12 13 9 7 9 14 15 16 15 14 9];  % 15 nodes

node = [7 3 2 1 2 3 7 4 5 6 5 4 7 9 10 9 13 12 11 12 13 9 14 15 16 15 14 9 7];  % 29 nodes

dim1 = 10;
dim2 = length(node)*3;
dim3 = 3;

shift_norm = 1;
scale_norm = 0;
rotation_norm = 1;
vel_norm = 1;
acc_norm = 0;

% %% train
% %[pose_4d_train, train_labels, ~, sequence_counter_train] = pose_tensor_formation_3Rotation(dim1,dim2, train_sequences, node, im_width, im_height, rotation_flag, vel_normalization, acc_normalization,dataset);
% [pose_4d_train, train_labels, ~, sequence_counter_train] = pose_tensor_formation_3Rotation_resize(dim1,dim2, train_sequences, node, rotation_flag, vel_normalization, acc_normalization,dataset);
% h5_file_train = sprintf(h5_file_train, sequence_counter_train);
% h5create(h5_file_train, '/dataset', size(pose_4d_train))  % create HDF5 data set
% h5write(h5_file_train, '/dataset', pose_4d_train)
% h5create(h5_file_train,'/label',size(train_labels),'Datatype','double');
% h5write(h5_file_train,'/label',train_labels);



%% test
%[pose_4d_test, test_labels, ~, sequence_counter_test] = pose_tensor_formation_3Rotation(dim1,dim2, test_sequences, node, rotation_flag, vel_normalization, acc_normalization,dataset);
[pose_4d_test, test_labels, ~, sequence_counter_test] = pose_tensor_formation_3Rotation_resize(dim1,dim2, dim3,test_sequences, node, shift_norm,scale_norm ,rotation_norm, vel_norm, acc_norm,dataset);
h5_file_test = sprintf(h5_file_test, sequence_counter_test);
h5create(h5_file_test, '/dataset', size(pose_4d_test))  % create HDF5 data set
h5write(h5_file_test, '/dataset', pose_4d_test)
h5create(h5_file_test,'/label',size(test_labels),'Datatype','double');
h5write(h5_file_test,'/label',test_labels);


