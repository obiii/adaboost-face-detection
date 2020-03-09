%% Hyper-parameters
%  You will need to change these. Start with a small number and increase
%  when your algorithm is working.

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;
% Number of weak classifiers
nbrWeakClassifiers = 90;

%% Load face and non-face data and plot a few examples
%  Note that the data sets are shuffled each time you run the script.
%  This is to prevent a solution that is tailored to specific images.

load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do NOT modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%Reference: https://engineering.purdue.edu/kak/Tutorials/AdaBoost.pdf

%weights for example
D = 1/length(xTrain) * ones(1, size(xTrain,2));
alpha = nan(nbrWeakClassifiers,1);
%errors = nan(length(xTrain), nbrHaarFeatures); % we need misrate of each thresh for each haar feature

error = Inf(nbrWeakClassifiers,1);
opt_p = nan(nbrWeakClassifiers,1);
opt_t = nan(nbrWeakClassifiers,1);
opt_f = nan(nbrWeakClassifiers,1);
%  Use your implementation of WeakClassifier and WeakClassifierError
for clsfr = 1:nbrWeakClassifiers % for all weak classifers
    clsfr
    %for each haar feature
    for feature = 1:nbrHaarFeatures
        
        X = xTrain(feature,:); %select one feature at a time
        Y = yTrain;
        
        sorted_X = sort(X); %We need ordered list, why? No idea yet, just follwoing the reference.
        
        %Treat each diiscrete value as decision threshold
        for i = 1:length(sorted_X);
           
           thres = sorted_X(i);
           p = 1; %polarity of 1 
           
           C = WeakClassifier(thres,p,X);
           E = WeakClassifierError(C,D,Y);
           
           if E >=0.5 % means classifier is no better than random guess
               p = -1;
               E = 1 -E;
           end
           
           if(E < error(clsfr,1))
               error(clsfr,1) = E;
               opt_p(clsfr,1) = p;
               opt_t(clsfr,1) = thres;
               opt_f(clsfr,1) = feature;
           end
        end
    end
    
    e_t =error(clsfr,1);
    alpha_t = (1/2).* log((1-e_t)/(e_t));
    alpha(clsfr,1)=alpha_t;
    
    h_x = WeakClassifier(opt_t(clsfr,1),opt_p(clsfr,1),xTrain(opt_f(clsfr,1),:));
    

    D_updated = D.* exp(-alpha_t.*yTrain.*h_x);
    D = D_updated / sum(D_updated);
    
end


[C,cM,acc] = strongClassify(nbrWeakClassifiers,alpha, opt_p,opt_f,opt_t,xTrain,yTrain);

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.


[C,cM,acc] = strongClassify(nbrWeakClassifiers,alpha, opt_p,opt_f,opt_t,xTest,yTest);
%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
accAll = nan(nbrWeakClassifiers,1);
for i = 1:nbrWeakClassifiers
    [C,cM,acc] = strongClassify(i,alpha, opt_p,opt_f,opt_t,xTest,yTest);
    accAll(i)=acc
end

plot(accAll)

%% Plot the error of the strong classifier as a function of the number of weak classifiers on train and Test Data


accAlltrain = nan(nbrWeakClassifiers,1);
accAlltest = nan(nbrWeakClassifiers,1);
for i = 1:nbrWeakClassifiers
    [C,cM,acc] = strongClassify(i,alpha, opt_p,opt_f,opt_t,xTrain,yTrain);
    accAlltrain(i)=acc;
end
for i = 1:nbrWeakClassifiers
    [C,cM,acc] = strongClassify(i,alpha, opt_p,opt_f,opt_t,xTest,yTest);
    accAlltest(i)=acc;
end

plot(accAlltrain)
hold on
plot(accAlltest)
xlabel("Number of weak classifiers");
ylabel("Accuracy of the strong classifier")
legend({'Train','Test'},'Location','southeast')
hold off

%% Plot some of the misclassified faces and non-faces from the test set
%  Use the subplot command to make nice figures with multiple images.
[C,cM,acc] = strongClassify(nbrWeakClassifiers,alpha, opt_p,opt_f,opt_t,xTest,yTest);

facedata = testImages(:,:,and(C == yTest,yTest ~= 1));
figure('Name','Missclassified as faces','NumberTitle','off');
colormap gray;

for k=1:16
    subplot(4,4,k), imagesc(facedata(:,:,k));
    axis image;
    axis off;
end


facedata = testImages(:,:,and(C == yTest,yTest ~= -1));

figure('Name','Missclassified as non-face','NumberTitle','off');
colormap gray;
for k=1:16
    subplot(4,4,k), imagesc(facedata(:,:,k));
    axis image;
    axis off;
end
%% Plot your choosen Haar-features
subplot


