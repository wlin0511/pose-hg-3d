%% Nomalization : Scaling + Shift + Rotation in xz plane + Rotation in xy plane + Rotation in yz plane

clear all
close all
addpath('/home/lin7lr/3Dpose/towards_3D_Human/src')  %%%%%%%%%%%%%

% filename = 'Ballfangen_catch_u_cm_np1_fr_goo_0';
classid = 1;   % 1, 2, 3, ... 12
switch classid
    case 1, classname = 'baseball_pitch';
    case 2, classname = 'clean_and_jerk';
    case 3, classname = 'pullup';
    case 4, classname = 'strum_guitar';
    case 5, classname = 'baseball_swing';
    case 6, classname = 'golf_swing';
    case 7, classname = 'pushup';
    case 8, classname = 'tennis_forehand';
    case 9, classname = 'bench_press';
    case 10, classname = 'jumping_jacks';
    case 11, classname = 'situp';
    case 12, classname = 'tennis_serve'; 
    case 13, classname = 'bowl';
    case 14, classname = 'jump_rope';
    case 15, classname = 'squat';
    otherwise, error('invalid classid!')
end

rgb_trainlist = fileread('/home/lin7lr/test_pose/list/Penn_action/rgb_trainlist.txt');
rgb_testlist = fileread('/home/lin7lr/test_pose/list/Penn_action/rgb_testlist.txt');
sequences = dir(['/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Penn_action/3DPose/', classname, '/*_', num2str(classid)]);
img_dir_prefix = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Penn_action/Penn_action_CroppedImg/';

dim1 = [];

visual_result_dir = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Intermediate_results/3DPoseVisualization/Pennaction_3Dpose/';
for sequence_id = 1:length(sequences)
    sequencePath = [sequences(sequence_id).folder, '/', sequences(sequence_id).name];
   
    close all
    result_dir = sequencePath;
    points_folderName = sequences(sequence_id).name;
    img_folderName = points_folderName(1:end-(1+length(num2str(classid))));
    img_dir = [img_dir_prefix, img_folderName];
    
    if ~isempty(strfind(rgb_trainlist, img_folderName))
        dataType = 'train';       
    elseif ~isempty(strfind(rgb_testlist, img_folderName))
        dataType = 'test';
    else
        error([img_folderName, ' is neither in trainlist nor in testlist!'])
    end
    myVideoName = [visual_result_dir, classname, '/',dataType, '/',points_folderName, '.avi'];

    all_imgs = dir([img_dir, '/*.png']);
    [~,img_order] = sort_nat({all_imgs.name});
    all_imgs = all_imgs(img_order);

    all_points = dir([result_dir, '/Pointsbbshift*.mat']);
    [~,points_order] = sort_nat({all_points.name});
    all_points = all_points(points_order);

    numImgs = size(all_imgs, 1);
    nKeyPoints = 16;
    LiWi = 3;
    edges = [1,2;2,3;3,7;7,4;4,5;5,6;11,12;12,13;13,9;9,14;14,15;15,16;7,9;9,10];
    colors = ['k','c','y','y','c','k','b','g','r','r','g','b','m','m'];
    %  colors = ['w','w','y','y','w','w','b','g','r','r','g','b','m','m'];
    rotation_flag = 1;

    fullpath_firstFramePoints = strcat(all_points(1).folder, '/', all_points(1).name);
    load(fullpath_firstFramePoints, 'x')
    firstFramePoints = x;
    %% normalization : scale
    firstFramePoints_scaled = firstFramePoints;
    firstFramePoints_scaled = firstFramePoints_scaled / 128;
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
    %% compute the rotation matrix inS001C001P001R001A001 yz-plane for all frames
    theta_yz = atan2(firstFramePoints_rotated_xy(13,3)- firstFramePoints_rotated_xy(7,3), firstFramePoints_rotated_xy(13,2)- firstFramePoints_rotated_xy(7,2) );
    if theta_yz < 0
       theta_yz = theta_yz + 2*pi;      %  theta :    0 to 2pi
    end
    % a counterclockwise rotation of (pi-theta_yz)
    R_yz = [cos(pi-theta_yz) -sin(pi-theta_yz); sin(pi-theta_yz) cos(pi-theta_yz)];


    if isempty(dim1)
        step = 1;
    else
        step = floor(numImgs/dim1);
    end

    if strcmp(dataType, 'train')
        fig1 = figure('Name', myVideoName,'NumberTitle','off', 'position', [50,300,400*5, 200*5]);
    else
        fig1 = figure('Name', myVideoName,'NumberTitle','off', 'position', [50,300,600*3, 400*1]);
    end
    set(gcf,'color','w');
    set(gca,'NextPlot','ReplaceChildren');
    clear frames
    for i = 1:step:numImgs
        fullpathImg = strcat(all_imgs(i).folder, '/', all_imgs(i).name);
    %     load(fullpathImg, 'x')
    %     img = permute(permute(x,[3,2,1]),[2,1,3]);
        img = imread(fullpathImg);
        fullpathPoints = strcat(all_points(i).folder, '/', all_points(i).name);
        load(fullpathPoints, 'x')
    %     points_original = x;
    %     points_original(:,2) = -points_original(:,2);
    %     points_original(:,3) = -points_original(:,3);
        points_original = x;


        %% normalization step1 : scale
        points_scaled = points_original/128;
        %% normalization step2 : shift according to the unified origin
        points_shifted = points_scaled;
        points_shifted = points_shifted - repmat( newOrigin, nKeyPoints, 1 );   % set the midpoint of keypoint 3 and keypoint 9 as the origin
        %% normalization step3 : rotation
        if rotation_flag
            points_rotated_xz = points_shifted;
    %         % a counterclockwise rotation of theta_xz
            points_rotated_xz(:,[1,3]) = (R_xz * (points_rotated_xz(:,[1,3]))')' ;
            points_rotated_xy = points_rotated_xz;
            points_rotated_xy(:, [1,2]) = (R_xy *(points_rotated_xy(:, [1,2]))')'  ;
            points_rotated_yz = points_rotated_xy;
            points_rotated_yz(:, [2,3]) = (R_yz *(points_rotated_yz(:, [2,3]))')'  ;

        end

        %% data augmentation based on rotation of different angles
        max_degree = 45;
        degree_step = 3;
        aug_degrees = -max_degree:degree_step:max_degree;   % rotation degrees in xz plane,   31 angles
        points_aug = zeros(size(points_rotated_yz,1),size(points_rotated_yz,2), length(aug_degrees));
        % for degreeid = 1:length(aug_degrees)
        for degreeid = [1,16,31]
            theta_aug = aug_degrees(degreeid)/180 * pi;

            R_xz_aug = [cos(theta_aug) -sin(theta_aug); sin(theta_aug) cos(theta_aug)];  % counterclockwise rotation of theta_aug
            points_rotated_aug = points_rotated_yz;
            points_rotated_aug(:,[1,3]) = (R_xz_aug * (points_rotated_aug(:,[1,3]))')' ;

            points_aug(:,:,degreeid) = points_rotated_aug;
        end

        %% 
        clf;
        if strcmp(dataType, 'train')
            hsub1 = subplot(2,3,1);
        else
            hsub1 = subplot(1,3,1);
        end
        imshow(img); hold on;
        if strcmp(dataType, 'train')  
            hsub2 = subplot(2,3,2);
        else
            hsub2 = subplot(1,3,2);
        end
        subplot_3DPose(points_original, nKeyPoints, edges, LiWi, colors )
%         xlim([50 200])
%         ylim([20 240])
%         zlim([100 200])
%         view([0,-20,-50])
        xlim auto
        ylim auto
        zlim auto
        view([0.3,-0.3,-0.1])
        set(gca,'Zdir','reverse')
        title('before normalization (scaling, shift and rotation)')
        if strcmp(dataType, 'train')
            hsub = subplot(2,3,3);
        else
            hsub = subplot(1,3,3);
        end
        subplot_3DPose(points_aug(:,:,16), nKeyPoints, edges, LiWi, colors )
%         xlim([-1 1])
%         ylim([-1 1])
%         zlim([-2 2])
        xlim auto
        ylim auto
        zlim auto
        view([-1,1,1])
        set(gca,'Zdir','reverse')
        title('degree 0')
        if strcmp(dataType, 'train')
            hsub = subplot(2,3,4);
            subplot_3DPose(points_aug(:,:,1), nKeyPoints, edges, LiWi, colors )
%             xlim([-0.5 0.5])
%             ylim([-0.6 1.2])
%             zlim([-0.5 0.5])
            xlim auto
            ylim auto
            zlim auto
            view([0,-0.5,-1])
            set(gca,'Zdir','reverse')
            %set(gca,'Ydir','reverse')
            title('aug -45')
            hsub = subplot(2,3,5);
            subplot_3DPose(points_aug(:,:,end), nKeyPoints, edges, LiWi, colors )
%             xlim([-0.5 0.5])
%             ylim([-0.6 1.2])
%             zlim([-0.5 0.5])
            xlim auto
            ylim auto
            zlim auto
            view([0,-0.5,-1])
            set(gca,'Zdir','reverse')
            %set(gca,'Ydir','reverse')
            title('aug 45')
        end
        pause(0.01);
        frames(i)=getframe(fig1);
    end
    if ~exist('JHMDB_train', 'dir')
        mkdir JHMDB_train
    end
    if ~exist('JHMDB_test', 'dir')
        mkdir JHMDB_test
    end
    
    if exist(myVideoName, 'file')
        eval(['delete ', myVideoName]) 
    end
    v = VideoWriter(myVideoName);
    v.Quality = 50;
    if isempty(dim1)
        v.FrameRate = 2;
    else  % dim1 = 10, only 10 frames are sampled
        v.FrameRate = 1; 
    end
    open(v);
    for i = 1:step:numImgs
        writeVideo(v,frames(i))
    end
    close(v);
    
end