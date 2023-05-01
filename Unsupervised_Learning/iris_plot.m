clc;
clear;

% High-dimensional flag
hd = false;

% tsne flag
ts = false;

% Distance
di = 'euclidean';

% Source
if hd == true
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [4 9 3 4];
    M = containers.Map(keys,values);
elseif ts == true
    % Distance-Cluster map
    keys = {'euclidean','manhattan','infinity','mahalanobis'};
    values = [2 4 2 14];
    M = containers.Map(keys,values);
else
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

%% Plot 1. K-means clustering

% Preprocess
pd = ai_preprocess(X,true);

% Number of clusters
c = M(di);

% Run clustering algorithm
[P,Y] = ai_kmeans(pd.X,c,0.01,di);

% Plot results
figure;
hold on;
grid on;
scatter3(pd.X(:,1),pd.X(:,2),pd.X(:,3),[],Y,'filled','MarkerEdgeColor','k');
scatter3(P(:,1),P(:,2),P(:,3),180,'filled','MarkerEdgeColor','k');
xlabel('sepal length');
ylabel('sepal width');
zlabel('petal length');
title('Iris Sample K-means Clustering with Euclidean Norm');
view(-137,10);

%% Plot 2. Fuzzy C-means clustering

% Preprocess
pd = ai_preprocess(X,true);

% Number of clusters
c = M(di);

% Run clustering algorithm
[P,Y] = ai_fcm(pd.X,c,2,0.01,di);

% Plot results
figure;
hold on;
grid on;
scatter3(pd.X(:,1),pd.X(:,2),pd.X(:,3),[],Y,'filled','MarkerEdgeColor','k');
scatter3(P(:,1),P(:,2),P(:,3),180,'filled','MarkerEdgeColor','k');
xlabel('sepal length');
ylabel('sepal width');
zlabel('petal length');
title('Iris Sample Fuzzy C-means Clustering with Euclidean Norm');
view(-137,10);

%% Plot 3. Hierarchical clustering

% Preprocess
pd = ai_preprocess(X,true);

% Number of clusters
c = M(di);

% Run clustering algorithm
[P,Y] = ai_hierarchical(pd.X,c,di);

% Plot results
figure;
hold on;
grid on;
scatter3(pd.X(:,1),pd.X(:,2),pd.X(:,3),[],Y,'filled','MarkerEdgeColor','k');
scatter3(P(:,1),P(:,2),P(:,3),180,'filled','MarkerEdgeColor','k');
xlabel('sepal length');
ylabel('sepal width');
zlabel('petal length');
title('Iris Sample Hierarchical Clustering with Euclidean Norm');
view(-137,10);