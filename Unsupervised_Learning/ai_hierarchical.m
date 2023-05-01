function [P,Y] = ai_hierarchical(X,c,dist)
%AI_HIERARCHICAL Hierarchical Clustering
%
%   Inputs:
%   X - Dataset
%   c - Number of centers
%   dist - Distance type
%
%   Outputs:
%   P - Prototypes / Centers
%   Y - Groups

% Covariance matrix
S = cov(X);

%% Step 1. Initialize clusters
n = size(X,1);
P = X;
U = logical(eye(n));

%% Iterate
while true
    %% Step 2. Compute centroid linkage clustering
    nc = size(P,1);
    if nc == c
        break;
    end
    L = ai_distmat(P,P,dist,S);
    L = triu(L);
    %% Step 3. Select clusters to merge
    minD = min(setdiff(L,min(L)));
    [c1,c2] = find(L == minD);
    c1 = c1(1);
    c2 = c2(1);
    %% Step 4. Merge clusters
    p = median([P(c1,:); P(c2,:)]);
    P(c1,:) = p;
    P(c2,:) = [];
    e = U(:,c2);
    U(e,c1) = true;
    U(:,c2) = [];
end

% Compute groups
[~,Y] = max(U,[],2);

end