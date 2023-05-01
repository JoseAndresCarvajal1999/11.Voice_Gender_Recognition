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
    dsrc = '..\data\iris_learning_hd.mat';
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [4 9 3 4];
    M = containers.Map(keys,values);
elseif ts == true
    % Datasource
    dsrc = '..\data\iris_learning_tsne.mat';
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [2 4 2 14];
    M = containers.Map(keys,values);
else
    % Datasource
    dsrc = '..\data\iris_learning_original.mat';
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [3 5 3 3];
    M = containers.Map(keys,values);
end

% Load dataset
load iris.dat;
X = iris;

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