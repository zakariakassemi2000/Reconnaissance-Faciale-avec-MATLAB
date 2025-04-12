clc;
clear;

load('faceClassifier.mat');  % Load the trained SVM
faceDetector = vision.CascadeObjectDetector();
cam = webcam();
pause(2); % Allow webcam to initialize

figure;
while true
    frame = snapshot(cam);
    bbox = step(faceDetector, frame);
    
    if ~isempty(bbox)
        for i = 1:size(bbox,1)
            face = imcrop(frame, bbox(i,:));
            gray = rgb2gray(face);
            gray = imresize(gray, [100 100]);
            hog = extractHOGFeatures(gray);
            label = predict(classifier, hog);
            
            frame = insertObjectAnnotation(frame, 'rectangle', bbox(i,:), label, 'Color', 'green');
        end
    end
    
    imshow(frame);
    drawnow;
end
