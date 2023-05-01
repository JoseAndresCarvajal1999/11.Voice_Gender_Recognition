function [P,Y] = ai_mountain(X,gg,sigma,beta,tol,dist)
%AI_MOUNTAIN Mountain Clustering
%
%   Inputs:
%   X - Dataset
%   gg - Gridding granularity
%   sigma - Application constant
%   beta - Substraction constant
%   tol - Tolerance
%   dist - Distance type
%
%   Outputs:
%   P - Prototypes / Centers
%   Y - Groups

% Prototypes
P = [];

% Last mountain
M_last = -Inf;

% Covariance matrix
S = cov(X);

%% Step 1. Setup grid
[n,d] = size(X);
V = {};
for i = 1:d
    V{i} = 0:gg:1;
end
V = cartesian(V{:});
l = size(V,1);
M = zeros(l,1);

%% Step 2. Build mountain function
D = ai_distmat(X,V,dist,S);
for k = 1:l
    s = 0;
    for i = 1:n
        s = s + exp(-(D(i,k)^2)/(2*sigma^2));
    end
    M(k) = s;
end

%% Iterate
while true
    %% Step 3. Select cluster center
    [~,I] = max(M);
    c_idx = I(1);
    c = V(c_idx,:);
    if abs(M_last - M(c_idx)) < tol
        break;
    end
    P = [P; c];
    M_last = M(c_idx);
    %% Step 4. Destruct mountain function
    d2c = ai_distmat(V,c,dist,S);
    M = M - M(c_idx)*exp(-(d2c.^2)/(2*beta^2));
end

% Compute groups
P = unique(P,'rows');
DP = ai_distmat(X,P,dist,S);
[~,Y] = min(DP,[],2);

end