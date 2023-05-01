clc;
clear;

% Debug flag
dbg = true;

% High-dimensional flag
hd = false;

% tsne flag
ts = false;

% Mountain clustering flag
mnt = false;

% Datasource
if hd == true
    dsrc = '..\data\voice_exploration_hd.mat';
elseif ts == true
    dsrc = '..\data\voice_exploration_tsne.mat';
else
    dsrc = '..\data\voice_exploration_original.mat';
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

% Exploration
[MR,SR] = ai_prototype_exploration(X,mnt,dbg);

% Save results
save(dsrc,'MR','SR');