clc;
clear;

% High-dimensional flag
hd = false;

% tsne flag
ts = false;

% Mountain clustering flag
mnt = true;

% Distance
di = 'euclidean';

% Datasource
if hd == true
    dsrc = '..\data\iris_exploration_hd.mat';
elseif ts == true
    dsrc = '..\data\iris_exploration_tsne.mat';
else
    dsrc = '..\data\iris_exploration_original.mat';
end

% Load exploration results
load(dsrc);

%% Analysis 1. Mountain Clustering
if mnt == true
    
    % Extract data
    dist = table2array(MR(:,1));
    MR = table2array(MR(:,2:end));
    
    % Plot indexes by number of clusters
    Inorm = (dist == di);
    CC = unique(MR(Inorm,3));
    Sil = grpstats(MR(Inorm,4),MR(Inorm,3),{'mean'});
    DB = grpstats(MR(Inorm,5),MR(Inorm,3),{'mean'});
    Pur = grpstats(MR(Inorm,6),MR(Inorm,3),{'mean'});
    Cal = grpstats(MR(Inorm,7),MR(Inorm,3),{'mean'});
    
    figure;
    hold on;
    plot(CC,Sil,'LineWidth',3,'DisplayName','Silhouette');
    plot(CC,DB,'LineWidth',3,'DisplayName','Davies-Bouldin');
    plot(CC,Pur,'LineWidth',3,'DisplayName','Purity');
    legend();
    title('Iris Mountain''s Indexes with Euclidean Norm');
    xlabel('number of clusters');
    
    figure;
    plot(CC,Cal,'LineWidth',3);
    title('Iris Mountain''s Calinski-Harabasz with Euclidean Norm');
    xlabel('number of clusters');
    ylabel('calinski-harabasz index');
    
    % Plot Silhouette index surface with parameters
    figure;
    tri = delaunay(MR(Inorm,1),MR(Inorm,2));
    trisurf(tri,MR(Inorm,1),MR(Inorm,2),MR(Inorm,4));
    shading interp;
    title('Iris Mountain''s Silhouette Index with Euclidean Norm');
    xlabel('\sigma');
    ylabel('\beta');
    zlabel('silhouette index');
    
    % Plot index surface with parameters
    figure;
    tri = delaunay(MR(Inorm,1),MR(Inorm,2));
    trisurf(tri,MR(Inorm,1),MR(Inorm,2),MR(Inorm,7));
    shading interp;
    title('Iris Mountain''s Calinski-Harabasz Index with Euclidean Norm');
    xlabel('\sigma');
    ylabel('\beta');
    zlabel('calinski-harabasz index');
    
end

%% Analysis 2. Subtractive Clustering

% Extract data
dist = table2array(SR(:,1));
SR = table2array(SR(:,2:end));

% Plot indexes by number of clusters
Inorm = (dist == di);
CC = unique(SR(Inorm,3));
Sil = grpstats(SR(Inorm,4),SR(Inorm,3),{'mean'});
DB = grpstats(SR(Inorm,5),SR(Inorm,3),{'mean'});
Pur = grpstats(SR(Inorm,6),SR(Inorm,3),{'mean'});
Cal = grpstats(SR(Inorm,7),SR(Inorm,3),{'mean'});

figure;
hold on;
plot(CC,Sil,'LineWidth',3,'DisplayName','Silhouette');
plot(CC,DB,'LineWidth',3,'DisplayName','Davies-Bouldin');
plot(CC,Pur,'LineWidth',3,'DisplayName','Purity');
legend();
title('Iris Subtractive''s Indexes with Euclidean Norm');
xlabel('number of clusters');

figure;
plot(CC,Cal,'LineWidth',3);
title('Iris Subtractive''s Calinski-Harabasz with Euclidean Norm');
xlabel('number of clusters');
ylabel('calinski-harabasz index');

% Plot indexes by parameter
Inorm = (dist == di);
R = unique(SR(Inorm,1));
Sil = grpstats(SR(Inorm,4),SR(Inorm,1),{'mean'});
DB = grpstats(SR(Inorm,5),SR(Inorm,1),{'mean'});
Pur = grpstats(SR(Inorm,6),SR(Inorm,1),{'mean'});
Cal = grpstats(SR(Inorm,7),SR(Inorm,1),{'mean'});

figure;
hold on;
plot(R,Sil,'LineWidth',3,'DisplayName','Silhouette');
plot(R,DB,'LineWidth',3,'DisplayName','Davies-Bouldin');
plot(R,Pur,'LineWidth',3,'DisplayName','Purity');
legend();
title('Iris Subtractive''s Indexes with Euclidean Norm');
xlabel('r_a');

figure;
plot(R,Cal,'LineWidth',3);
title('Iris Subtractive''s Calinski-Harabasz with Euclidean Norm');
xlabel('r_a');
ylabel('calinski-harabasz index');