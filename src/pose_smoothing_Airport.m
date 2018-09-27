close all
addpath('/home/lin7lr/3Dpose/towards_3D_Human/src')
addpath('/home/lin7lr/3Dpose/towards_3D_Human/src/pSplineSmoothing')


%   5734_17_1 in visual video     5734_16_1 in sequences

dataset = 'Airport';
if strcmp(dataset, 'Airport')
    poseSequence_5734train = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_5734/train/SAM*');
    poseSequence_5735train = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_5735/train/SAM*');
    poseSequence_5734test = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_5734/test/SAM*');
    poseSequence_5735test = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_5735/test/SAM*');
    test_sequences_Laptop5 = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_Laptop5_S2/test/Laptop*');
    test_sequences_Laptop1 = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_Laptop1/test/Laptop*');
    test_sequences_Laptop6 = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_Laptop6/test/Laptop*');
    test_sequences_Laptop6_S1 = dir('/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_Laptop6_S1/test/Laptop*');
    Result_dir = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/pSplineSmoothed/Pose3D_Laptop6_S1/test';
    
    sequences = test_sequences_Laptop6_S1;
    
elseif strcmp(dataset, 'JHMDB')
    trainlist_rgb = '/home/lin7lr/test_pose/list/JHMDB/train_test_subsplit2/rgb_trainlist_subsplit2.txt';
    testlist_rgb = '/home/lin7lr/test_pose/list/JHMDB/train_test_subsplit2/rgb_testlist_subsplit2.txt';

    sequences = get_sequence_paths(testlist_rgb, dataset, 0, 0); 
elseif strcmp(dataset, 'Penn_action')
    trainlist_rgb = '/home/lin7lr/test_pose/list/Penn_action/rgb_trainlist.txt';
    testlist_rgb = '/home/lin7lr/test_pose/list/Penn_action/rgb_testlist.txt';
    sequences = get_sequence_paths(testlist_rgb,dataset);
end

nKeyPoints = 16;

options = fitoptions('Method','Smooth','SmoothingParam',0.07);

smoothing_flag = 'pspline';


Ni = 1;
tol = 0.001;
s = 0.07;
lambda = min(s,1);
numKnots = 8;
bdeg = 3;
pord = 2;
variableSmoothingScale1D = ones(1, numKnots + bdeg - pord);


%sequenceid = 1;
function smoothing_sequences()

    for sequenceid = 1:length(sequences)
        if strcmp(dataset, 'JHMDB') || strcmp(dataset, 'Penn_action')
            sequencePath = sequences{sequenceid};
        else
            sequencePath = [sequences(sequenceid).folder, '/', sequences(sequenceid).name];
            strcell_array = strsplit(sequencePath, '_');
            classid = str2num(strcell_array{end});
            objectid = str2num(strcell_array{end-3});
        end
        disp(['smoothing ', sequencePath])

        allMat = dir([sequencePath,'/Pointsbbshift*.mat']);
        [~,matOrder] = sort_nat({allMat.name});
        allMat = allMat(matOrder);
        numFrames = length(allMat);

        pose_cube = zeros(nKeyPoints, 3, numFrames);
        pose_cube_smoothed = zeros(nKeyPoints, 3, numFrames);    
        %% load the original pose data into  pose_cube
        for frameid = 1:numFrames
            Points_path = [allMat(frameid).folder,'/', allMat(frameid).name];
            load(Points_path, 'x')
            pose_cube(:,:,frameid) = x;
        end

        %%  smoothing
        if strcmp(smoothing_flag, 'smooth')    
            for keypointid = 1:nKeyPoints
                for col_index = 1:3   %  x, y, z
                    f = fit((1:numFrames)', reshape(pose_cube(keypointid,col_index,:),numFrames,1),'smooth', options );
                    pose_cube_smoothed(keypointid,col_index,:) = f((1:numFrames)');
                end
            end
        elseif  strcmp(smoothing_flag, 'pspline')  
            z_S21 = 1:numFrames;
            gridWeight = ones(1, numFrames);
            pSplineFunc = @(dyb,w) pSplineSmoothingEquiDistWeighted1D(z_S21, dyb, w, variableSmoothingScale1D, numKnots, bdeg, pord, s);   
            for keypointid = 1:nKeyPoints
                dataBeforeSmoo_x = reshape(pose_cube(keypointid,1,:),1,numFrames);
                dataBeforeSmoo_y = reshape(pose_cube(keypointid,2,:),1,numFrames);
                dataBeforeSmoo_z = reshape(pose_cube(keypointid,3,:),1,numFrames);

                [dataAfterSmoo_x, ~, ~, ~, ~] = l1Regression1D(pSplineFunc, dataBeforeSmoo_x, [], gridWeight, lambda, tol, Ni);
                [dataAfterSmoo_y, ~, ~, ~, ~] = l1Regression1D(pSplineFunc, dataBeforeSmoo_y, [], gridWeight, lambda, tol, Ni);
                [dataAfterSmoo_z, ~, ~, ~, ~] = l1Regression1D(pSplineFunc, dataBeforeSmoo_z, [], gridWeight, lambda, tol, Ni);
                pose_cube_smoothed(keypointid,1,:) = dataAfterSmoo_x;
                pose_cube_smoothed(keypointid,2,:) = dataAfterSmoo_y;
                pose_cube_smoothed(keypointid,3,:) = dataAfterSmoo_z;

            end       
        end   
        for frameid = 1:numFrames
            x = pose_cube_smoothed(:,:, frameid );
            if strcmp(dataset, 'Airport')
                data_dir = [Result_dir,'/',sequences(sequenceid).name];
                mat_filename = [ data_dir, '/Points',num2str(frameid),'.mat' ];
            elseif strcmp(dataset, 'JHMDB')
                data_dir = strrep(sequencePath, '3DPose', '3DPose_pSplineSmoothed');
                mat_filename = [ data_dir, '/Pointsbbshift',sprintf('%05d', frameid),'.mat' ];
            elseif strcmp(dataset, 'Penn_action')
                data_dir = strrep(sequencePath, '3DPose', '3DPose_pSplineSmoothed');
                mat_filename = [ data_dir, '/Points',sprintf('%06d', frameid),'.mat' ];
            end
            if ~exist(data_dir, 'dir')
               mkdir(data_dir) 
            end
            save( mat_filename, 'x' )
        end
    end

end







