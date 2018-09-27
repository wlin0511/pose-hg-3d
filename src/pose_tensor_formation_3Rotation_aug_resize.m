function [pose_4d, labels, sequence_id, sequence_counter] = ...
    pose_tensor_formation_3Rotation_aug_resize(dim1,dim2,dim3, sequences, node,  rotation_flag, vel_normalization, acc_normalization, dataset, selected_classes)
    % pose_4d = zeros(dim1, dim2, 3, length(sequences));
    % labels = zeros(length(sequences), 1);
    nKeyPoints = 16;
    sequence_counter = 0;
    %% data augmentation based on rotation of different angles
    max_degree = 45;
    degree_step = 3;
    aug_degrees = -max_degree:degree_step:max_degree;   % rotation degrees in xz plane,   31 angles
    numAngles = length(aug_degrees);
    for sequenceid = 1:length(sequences)
        if strcmp(dataset, 'JHMDB')|| strcmp(dataset, 'Penn_action')
            sequencePath = sequences{sequenceid};
        else
            sequencePath = [sequences(sequenceid).folder, '/', sequences(sequenceid).name];
        end
        % labels(sequenceid,1) = str2double(sequencePath(end));
        disp(['processing ', sequencePath])
        allMat = dir([sequencePath,'/Pointsbbshift*.mat']);
        [~,matOrder] = sort_nat({allMat.name});
        allMat = allMat(matOrder);
        numFrames = length(allMat);
        if numFrames >= dim1
            sequence_counter = sequence_counter + 1;
            % error([sequencePath, ' has only ', num2str(numFrames), ' frames'])
            step = floor(numFrames/dim1);
            ticks = 1:step:min(step*dim1, numFrames);
            pose_tensor_aug = zeros(dim1, dim2, dim3, numAngles);            
            fullpath_firstFramePoints = [allMat(ticks(1)).folder,'/', allMat(ticks(1)).name];
            load(fullpath_firstFramePoints, 'x')
            firstFramePoints = x;
            %% normalization: scale
            firstFramePoints_scaled = x;
            %% compute the unified origin for all frames
            newOrigin = (firstFramePoints_scaled(7,:) + firstFramePoints_scaled(9,:))/2;         
            firstFramePoints_shifted = firstFramePoints_scaled;
            firstFramePoints_shifted = firstFramePoints_shifted - repmat( newOrigin, nKeyPoints, 1 );
            %% compute the rotation matrix in xz-plane for all frames
            theta_xz = atan2(firstFramePoints_shifted(14,3)- firstFramePoints_shifted(13,3), firstFramePoints_shifted(14,1)- firstFramePoints_shifted(13,1) );
            if theta_xz < 0
               theta_xz = theta_xz + 2*pi;      %  theta :    0 to 2pi
            end
            % a clockwise rotation of theta_xz (around the origin, which is the midpoint of keypoint7 and keypoint9)
            % a clockwise rotation of theta_xz = a counterclockwise rotation of -theta_xz
            R_xz = [cos(-theta_xz) -sin(-theta_xz); sin(-theta_xz) cos(-theta_xz)];         
            firstFramePoints_rotated_xz = firstFramePoints_shifted;
            firstFramePoints_rotated_xz(:, [1,3]) =  (R_xz * (firstFramePoints_rotated_xz(:, [1,3]))')' ;
            %% compute the rotation matrix in xy-plane for all frames
            theta_xy = atan2(firstFramePoints_rotated_xz(14,2)- firstFramePoints_rotated_xz(13,2), firstFramePoints_rotated_xz(14,1)- firstFramePoints_rotated_xz(13,1) );
            if theta_xy < 0
               theta_xy = theta_xy + 2*pi;      %  theta :    0 to 2pi
            end
            % a clockwise rotation of theta_xy = a counterclockwise rotation of -theta_xy
            R_xy = [cos(-theta_xy) -sin(-theta_xy); sin(-theta_xy) cos(-theta_xy)];           
            firstFramePoints_rotated_xy = firstFramePoints_rotated_xz;
            firstFramePoints_rotated_xy(:, [1,2]) =  (R_xy * (firstFramePoints_rotated_xy(:, [1,2]))')' ;
            %% compute the rotation matrix in yz-plane for all frames
            theta_yz = atan2(firstFramePoints_rotated_xy(13,3)- firstFramePoints_rotated_xy(7,3), firstFramePoints_rotated_xy(13,2)- firstFramePoints_rotated_xy(7,2) );
            if theta_yz < 0
               theta_yz = theta_yz + 2*pi;      %  theta :    0 to 2pi
            end
            % a counterclockwise rotation of (pi-theta_yz)
            R_yz = [cos(pi-theta_yz) -sin(pi-theta_yz); sin(pi-theta_yz) cos(pi-theta_yz)];    
            
            
            
            joint_positions_aug = zeros(numFrames, dim2, 1, numAngles);
            for frameid = 1:numFrames
                matPath = [allMat(frameid).folder,'/', allMat(frameid).name];
                load(matPath, 'x')                
                %% normalization step1 : scale
                points_scaled = x;
                %% normalization step2 : shift according to the unified origin
                points_shifted = points_scaled;
                points_shifted = points_shifted - repmat( newOrigin, nKeyPoints, 1 );   % set the midpoint of keypoint 3 and keypoint 9 as the origin
                %% normalization step3 : rotation
                if 1
                    points_rotated_xz = points_shifted;
                    % a counterclockwise rotation of theta_xz
                    points_rotated_xz(:,[1,3]) = (R_xz * (points_rotated_xz(:,[1,3]))')' ;
                    points_rotated_xy = points_rotated_xz;
                    points_rotated_xy(:, [1,2]) = (R_xy *(points_rotated_xy(:, [1,2]))')'  ;
                    points_rotated_yz = points_rotated_xy;
                    points_rotated_yz(:, [2,3]) = (R_yz *(points_rotated_yz(:, [2,3]))')'  ;
                end
                for degreeid = 1:numAngles % -45 ~ 45,  step 3,  31 angles
                    theta_aug = aug_degrees(degreeid)/180 * pi;
                    R_xz_aug = [cos(theta_aug) -sin(theta_aug); sin(theta_aug) cos(theta_aug)];  % counterclockwise rotation of theta_aug
                    points_rotated_aug = points_rotated_yz;
                    points_rotated_aug(:,[1,3]) = (R_xz_aug * (points_rotated_aug(:,[1,3]))')' ;
                    xyz_nodeSequence = points_rotated_aug(node,:)';
                    joint_positions_aug(frameid, :, 1, degreeid) = xyz_nodeSequence(:)';
                end
                
            end
            %joint_positions_aug_resized = zeros(dim1, dim2, 1, numAngles);
            for degreeid = 1:numAngles
                %joint_positions_aug_resized(:,:,1,degreeid) = imresize(joint_positions_aug(:,:,1,degreeid), [dim1, dim2]);
                pose_tensor_aug(:,:,1,degreeid) = imresize(joint_positions_aug(:,:,1,degreeid), [dim1, dim2]);
                if dim3 == 2
                    for dim1_index = 2:dim1
                            if vel_normalization
                                pose_tensor_aug(dim1_index,:,2,degreeid) = (pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid))/step;
                            else
                                pose_tensor_aug(dim1_index,:,2,degreeid) = pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid);
                            end
                    end
                elseif dim3 == 3
                    for dim1_index = 2:dim1
                            if vel_normalization
                                pose_tensor_aug(dim1_index,:,2,degreeid) = (pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid))/step;
                            else
                                pose_tensor_aug(dim1_index,:,2,degreeid) = pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid);
                            end
                            if acc_normalization
                                pose_tensor_aug(dim1_index,:,3,degreeid) = (pose_tensor_aug(dim1_index,:,2,degreeid) - pose_tensor_aug(dim1_index-1, :,2,degreeid))/step;
                            else
                                pose_tensor_aug(dim1_index,:,3,degreeid) = pose_tensor_aug(dim1_index,:,2,degreeid) - pose_tensor_aug(dim1_index-1, :,2,degreeid);
                            end
                    end                    
                end
            end             
%             for dim1_index = 1: dim1     % for i = 1:10
%                 matid = ticks(dim1_index);
%                 matPath = [allMat(matid).folder,'/', allMat(matid).name];
%                 load(matPath, 'x')
%                 points_original = x;
%                 %% normalization step1 : scale
%                 points_scaled = x;
%                 %% normalization step2 : shift according to the unified origin
%                 points_shifted = points_scaled;
%                 points_shifted = points_shifted - repmat( newOrigin, nKeyPoints, 1 );   % set the midpoint of keypoint 3 and keypoint 9 as the origin
%                 %% normalization : rotation
%                 %% normalization step3 : rotation
%                 if rotation_flag
%                     points_rotated_xz = points_shifted;
%                     % a counterclockwise rotation of theta_xz
%                     points_rotated_xz(:,[1,3]) = (R_xz * (points_rotated_xz(:,[1,3]))')' ;
%                     points_rotated_xy = points_rotated_xz;
%                     points_rotated_xy(:, [1,2]) = (R_xy *(points_rotated_xy(:, [1,2]))')'  ;
%                     points_rotated_yz = points_rotated_xy;
%                     points_rotated_yz(:, [2,3]) = (R_yz *(points_rotated_yz(:, [2,3]))')'  ;
%                 end               
%                 for degreeid = 1:numAngles   % -45 ~ 45,  step 3,  31 angles
%                     theta_aug = aug_degrees(degreeid)/180 * pi;
%                     R_xz_aug = [cos(theta_aug) -sin(theta_aug); sin(theta_aug) cos(theta_aug)];  % counterclockwise rotation of theta_aug
%                     points_rotated_aug = points_rotated_yz;
%                     points_rotated_aug(:,[1,3]) = (R_xz_aug * (points_rotated_aug(:,[1,3]))')' ;
%                     xyz_nodeSequence = points_rotated_aug(node,:)';    
%                     pose_tensor_aug(dim1_index, :, 1,degreeid) = xyz_nodeSequence(:)';
%                     if dim1_index ~= 1   % if i ==1, the first and second derivatives are 0
%                         if vel_normalization
%                             pose_tensor_aug(dim1_index,:,2,degreeid) = (pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid))/step;
%                         else
%                             pose_tensor_aug(dim1_index,:,2,degreeid) = pose_tensor_aug(dim1_index,:,1,degreeid) - pose_tensor_aug(dim1_index-1, :,1,degreeid);
%                         end
%                         if acc_normalization
%                             pose_tensor_aug(dim1_index,:,3,degreeid) = (pose_tensor_aug(dim1_index,:,2,degreeid) - pose_tensor_aug(dim1_index-1, :,2,degreeid))/step;
%                         else
%                             pose_tensor_aug(dim1_index,:,3,degreeid) = pose_tensor_aug(dim1_index,:,2,degreeid) - pose_tensor_aug(dim1_index-1, :,2,degreeid);
%                         end
%                     end                 
%                 end  % for degreeid = 1:length(aug_degrees)            
%             end  % for dim1_index = 1: dim1         
            if strcmp(dataset, 'JHMDB')|| strcmp(dataset, 'Penn_action')
                strcell_array =  strsplit(sequencePath, '_');
                classid = str2num(strcell_array{end});
                sequence_id = [];
                if exist('selected_classes', 'var')
                    new_classid = find(selected_classes == classid);
                    labels((sequence_counter-1)*numAngles+1 : sequence_counter*numAngles ,  1) = new_classid;
                else
                    labels((sequence_counter-1)*numAngles+1 : sequence_counter*numAngles ,  1) = classid;
                end
                
            else
                strcell_array =  strsplit(sequences(sequenceid).name, '_');
                labels( (sequence_counter-1)*numAngles+1 : sequence_counter*numAngles ,  1) = str2num(strcell_array{end});
                str_position = find(ismember(strcell_array, 'id'));
                sequence_id((sequence_counter-1)*numAngles+1 : sequence_counter*numAngles, : ) = repmat([str2num(strcell_array{str_position+1}), str2num(strcell_array{str_position+2})], numAngles, 1);
            end
            pose_4d(:,:,:,(sequence_counter-1)*numAngles+1 : sequence_counter*numAngles) = pose_tensor_aug;
        else  % numFrames < dim1
            disp([sequencePath, ' has only ', num2str(numFrames), ' frames'])
        end  % if numFrames >= dim1
    end  % sequenceid = 1:length(sequences)
end