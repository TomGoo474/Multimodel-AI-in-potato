%Train Vision  Network for Image Classification
%load  your image folder
imageFolder = 'F:\XX\XX\';

% Create an image datastore containing the images.
imds = imageDatastore(imageFolder,IncludeSubfolders=true,LabelSource="foldernames");
% View the number of classes.
classNames = categories(imds.Labels);
numClasses = numel(categories(imds.Labels))

% Split the datastore into training, validation, and test partitions using the splitEachLabel function. Use 80% of the images for training and set aside 10% for validation and 10% for testing.
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.8,0.1);
%Batchsize set 16
miniBatchSize = 16;
numObservationsTrain = numel(augimdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);

%load vision models structure
load trainingmodels.mat
%define the cell for saving trained models
completed_trained_Net = cell(14,1);
%define the cell for saving test accuracy
Test_acc = cell(14,1);

% Train Neural Network
% Train the neural network using the trainnet function. For classification, use cross-entropy loss. By default, the trainnet function uses a GPU if one is available. Training on a GPU requires a Parallel Computing Toolbox™ license and a supported GPU device. For information on supported devices, see GPU Computing Requirements. Otherwise, the trainnet function uses the CPU. To specify the execution environment, use the ExecutionEnvironment training option.
% This example trains the network using an NVIDIA Titan RTX GPU with 24 GB RAM. The training takes about 37 minutes to run.

for s = 1:14

        switch s
            case 1
                %load VGG
                inputSize = [227 227 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                
            case 2
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net =Trainingmodels{s,1};
                %load Alexnet;
            case 3
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load GoogleNet;
            case 4
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load ShuffleNet;

            case 5
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load DenseNet201;
            case 6
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load ResNet18;
            case 7
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                % load ResNet50;
            case 8
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1}; 
                %load ResNet101;
            case 9
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load MobileNetv2;
            case 10
                inputSize = [224 224 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load EfficientNetb0;
            case 11

                %change the image size
                inputSize = [299 299 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load Inceptionv3;
               
            case 12
                inputSize = [299 299 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load Xception;
               
            case 13
                inputSize = [384 384 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
                %load transformer;
               
           case 14
                 % load MSC-ResViT
               inputSize = [384 384 3];
                augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
                    ColorPreprocessing="gray2rgb");
                augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
                    ColorPreprocessing="gray2rgb");
                net = Trainingmodels{s,1};
               
                
        
        end
        %define trained options
        options = trainingOptions("adam", ...
        MaxEpochs=1, ...
        InitialLearnRate=0.0001, ...
        MiniBatchSize=miniBatchSize, ...
        ValidationData=augimdsValidation, ...
        ValidationFrequency=numIterationsPerEpoch, ...
        OutputNetwork="auto", ...
        Plots="training-progress", ...
        Metrics="accuracy", ...
        ExecutionEnvironment="gpu",...
                Verbose=false);
        %training Model
        completed_trained_Net{s,1} = trainnet(augimdsTrain,net,"crossentropy",options);
         YTest = minibatchpredict(completed_trained_Net{s,1},augimdsTest);
         YTest = onehotdecode(YTest,classNames,2);
        % figure
        % TTest = imdsTest.Labels;
        % confusionchart(TTest,YTest)
        Test_acc{s,1} = mean(YTest == TTest);
end

%plot P-R curve,this section you should change the image size under
%different deep learning models

for i = 1:14

        YPred = minibatchpredict(completed_trained_Net{i,1},augimdsTest);

        % Decode the one-hot encoded predictions
        YTest = onehotdecode(YPred, classNames, 2);
        
        % Convert labels to categorical if not already
        TTest = categorical(imdsTest.Labels);
        
        % Compute the predicted scores for each class
        numClasses = numel(classNames);
        scores = YPred; % Assuming YPred contains the scores for each class
        
        % Binarize the true labels for one-vs-all approach
        trueLabels = zeros(numel(TTest), numClasses);
        for k = 1:numClasses
            trueLabels(:, k) = (TTest == classNames(k));
        end
        
        % Concatenate all scores and true labels for overall P-R curve
        allScores = scores(:);
        allTrueLabels = trueLabels(:);
        
        % Compute precision and recall
        [recall, precision, ~] = perfcurve(allTrueLabels, allScores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
        
        % Plot Precision-Recall curve
        
        plot(recall, precision);
        hold on

end

