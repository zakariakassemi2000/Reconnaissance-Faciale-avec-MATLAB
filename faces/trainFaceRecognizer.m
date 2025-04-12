clc;
clear;

faceDetector = vision.CascadeObjectDetector();
imageFolder = 'dataset'; % change to your path
people = dir(imageFolder);
features = [];
labels = [];

for i = 3:length(people) % skip '.' and '..'
    personName = people(i).name;
    personPath = fullfile(imageFolder, personName);
    images = dir(fullfile(personPath, '*.jpg'));
    
    for j = 1:length(images)
        imgPath = fullfile(personPath, images(j).name);
        img = imread(imgPath);
        
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        
        bbox = step(faceDetector, img);
        if ~isempty(bbox)
            face = imcrop(img, bbox(1,:));
            face = imresize(face, [100 100]);
            hog = extractHOGFeatures(face);
            features = [features; hog];
            labels = [labels; {personName}];
        end
    end
end

% Train classifier
classifier = fitcecoc(features, labels);

% Save model
save('faceClassifier.mat', 'classifier');
disp("Model trained and saved.");
