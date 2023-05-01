clear;
clc;

% Debug flag
dbg = true;

% High-dimensional flag
hd = false;

% tsne flag
ts = false;

% Source
if hd == true
    % Datasource
    dsrc = '..\data\voice_learning_hd.mat';
    % Distance-Cluster map
    keys = {'euclidean','infinity'};
    values = [6 2];
    M = containers.Map(keys,values);
elseif ts == true
    % Datasource
    dsrc = '..\data\voice_learning_tsne.mat';
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [16 32 13 33];
    M = containers.Map(keys,values);
else
    % Datasource
    dsrc = '..\data\voice_learning_original.mat';
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity'};
    values = [9 2 4];
    M = containers.Map(keys,values);
end

% Load dataset
X = readmatrix('..\data\voice.csv');

% Augment feature space
if hd == true
    Y = X(:,end);
    X = X(:,1:end-1);
    X = x2fx(X,'quadratic');
    X = [X(:,2:end) Y];
end

% Apply embedding
if ts == true
    Y = X(:,end);
    X = X(:,1:end-1);
    X = tsne(X);
    X = [X(:,1:end) Y];
end

% Learning
[KMR,FCR,HCR] = ai_prototype_learning(X,M,dbg);

% Save results
save(dsrc,'KMR','FCR','HCR');