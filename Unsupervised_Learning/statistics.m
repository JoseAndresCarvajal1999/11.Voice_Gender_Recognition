clear;
clc;

% Load dataset
X = readmatrix('..\data\voice.csv');
y = X(:,end);
X = X(:,1:end-1);

% One-way analysis of variance (ANOVA)
[p,t,stats] = anova1(X(:,1),y,'off');

% Perform a multiple comparison of the group means
[c,m,h,nms] = multcompare(stats);

% Select male
Xmale = X(y==1,:);

% Split male group in 2
n_male = size(Xmale,1);
h_male = n_male/2;
Xmale1 = Xmale(1:h_male,1);
Xmale2 = Xmale(h_male:end-1,1);

% Kruskal-Wallis test
p_male = kruskalwallis([Xmale1 Xmale2]);
title('Box Plot of Male Frequencies (Upper and Lower Halves)');
ylabel('mean frequency (Hz)');

% Select female
Xfemale = X(y==0,:);

% Split female group in 2
n_female = size(Xfemale,1);
h_female = n_female/2;
Xfemale1 = Xfemale(1:h_female,1);
Xfemale2 = Xfemale(h_female:end-1,1);

% Kruskal-Wallis test
p_female = kruskalwallis([Xfemale1 Xfemale2]);
title('Box Plot of Female Frequencies (Upper and Lower Halves)');
ylabel('mean frequency (Hz)');