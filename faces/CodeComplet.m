clc;
clear;


disp('⏳ Training the model...');

faceDetector = vision.CascadeObjectDetector();
imageFolder = 'dataset'; %dataset path
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
disp("✅ Model trained and saved.");

disp('🎥 Starting real-time face recognition...');

load('faceClassifier.mat');  % Load the trained model
cam = webcam();
pause(2); % Let webcam initialize

figure;
while true
    frame = snapshot(cam);
    bbox = step(faceDetector, frame);
    
    if ~isempty(bbox)
        for i = 1:size(bbox,1)
            face = imcrop(frame, bbox(i,:));
            
            if size(face,3) == 3
                gray = rgb2gray(face);
            else
                gray = face;
            end
            
            gray = imresize(gray, [100 100]);
            hog = extractHOGFeatures(gray);
            label = predict(classifier, hog);
            
            frame = insertObjectAnnotation(frame, 'rectangle', bbox(i,:), label, 'Color', 'green');
        end
    end
    
    imshow(frame);
    title('Live Face Recognition');
    drawnow;
end
