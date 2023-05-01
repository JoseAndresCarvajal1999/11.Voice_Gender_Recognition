clc;
clear;

% High-dimensional flag
hd = false;

% tsne flag
ts = false;

% Datasource
if hd == true
    dsrc = '..\data\iris_learning_hd.mat';
elseif ts == true
    dsrc = '..\data\iris_learning_tsne.mat';
else
    dsrc = '..\data\iris_learning_original.mat';
end

% Load learning results
load(dsrc);

%% Analysis 1. K-means

% Silhouette by distance
[kSi,kSiDist] = grpstats(KMR.Silhouettes,KMR.Distance,{'mean','gname'});

% Davies-Bouldin by distance
[kDb,kDbDist] = grpstats(KMR.DBIndex,KMR.Distance,{'mean','gname'});

% Purity by distance
[kPu,kPuDist] = grpstats(KMR.Purity,KMR.Distance,{'mean','gname'});

% Calinski-Harabasz by distance
[kCa,kCaDist] = grpstats(KMR.CalinskiHarabasz,KMR.Distance,{'mean','gname'});

%% Analysis 2. Fuzzy C-means

% Silhouette by distance
[fSi,fSiDist] = grpstats(FCR.Silhouettes,FCR.Distance,{'mean','gname'});

% Davies-Bouldin by distance
[fDb,fDbDist] = grpstats(FCR.DBIndex,FCR.Distance,{'mean','gname'});

% Purity by distance
[fPu,fPuDist] = grpstats(FCR.Purity,FCR.Distance,{'mean','gname'});

% Calinski-Harabasz by distance
[fCa,fCaDist] = grpstats(FCR.CalinskiHarabasz,FCR.Distance,{'mean','gname'});

%% Analysis 3. Hierarchical clustering

% Silhouette by distance
[hSi,hSiDist] = grpstats(HCR.Silhouettes,HCR.Distance,{'mean','gname'});

% Davies-Bouldin by distance
[hDb,hDbDist] = grpstats(HCR.DBIndex,HCR.Distance,{'mean','gname'});

% Purity by distance
[hPu,hPuDist] = grpstats(HCR.Purity,HCR.Distance,{'mean','gname'});

% Calinski-Harabasz by distance
[hCa,hCaDist] = grpstats(KMR.CalinskiHarabasz,KMR.Distance,{'mean','gname'});