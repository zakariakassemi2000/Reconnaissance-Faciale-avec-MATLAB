function FaceRecognitionGUI
    % Créer la figure principale
    fig = uifigure('Name', 'Face Recognition GUI', 'Position', [100 100 600 500]);

    % Zone d'affichage pour la vidéo
    ax = uiaxes(fig, 'Position', [50 100 500 350]);
    title(ax, 'Live Face Recognition');
    axis(ax, 'off');

    % Bouton pour entraîner le modèle
    btnTrain = uibutton(fig, 'push', 'Text', '📚 Entraîner le modèle', ...
        'Position', [50, 30, 200, 40], ...
        'ButtonPushedFcn', @(btn,event)trainModel());

    % Bouton pour démarrer la reconnaissance
    btnStart = uibutton(fig, 'push', 'Text', '🎥 Démarrer la reconnaissance', ...
        'Position', [300, 30, 200, 40], ...
        'ButtonPushedFcn', @(btn,event)startRecognition(ax));

end

function trainModel()
    clc; clearvars -except fig;  
    disp('⏳ Entraînement du modèle...');
    
    faceDetector = vision.CascadeObjectDetector();
    imageFolder = 'dataset'; % chemin vers les images
    people = dir(imageFolder);
    features = [];
    labels = [];

    for i = 3:length(people) % ignorer '.' et '..'
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

    classifier = fitcecoc(features, labels);
    save('faceClassifier.mat', 'classifier');
    uialert(gcf, '✅ Modèle entraîné et sauvegardé.', 'Succès');
end

function startRecognition(ax)
    load('faceClassifier.mat'); % Charger le modèle
    faceDetector = vision.CascadeObjectDetector();
    cam = webcam();
    pause(2);

    while isvalid(ax) % Tant que la figure n’est pas fermée
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

        imshow(frame, 'Parent', ax);
        drawnow;
    end
end
