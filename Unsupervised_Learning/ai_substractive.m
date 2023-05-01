function [P,Y] = ai_substractive(X,ra,rb,tol,dist)
%AI_SUBSTRACTIVE Substractive Clustering
%
%   Inputs:
%   X - Dataset
%   ra - Neighborhood radius
%   rb - Measurable reduction
%   tol - Tolerance
%   dist - Distance type
%
%   Outputs:
%   P - Prototypes / Centers
%   Y - Groups

% Prototypes
P = [];

% Last density
De_last = -Inf;

% Covariance matrix
S = cov(X);

%% Step 1. Calculate density measures
n = size(X,1);
D = ai_distmat(X,X,dist,S);
De = zeros(n,1);
for i = 1:n
    s = 0;
    for j = 1:n
        s = s + exp(-(D(i,j)^2)/((ra/2)^2));
    end
    De(i) = s;
end

%% Iterate
while true
    %% Step 2. Select cluster center
    [~,I] = max(De);
    c_idx = I(1);
    c = X(c_idx,:);
    if abs(De_last - De(c_idx)) < tol
        break;
    end
    P = [P; c];
    De_last = De(c_idx);
    %% Step 3. Update density measures
    De = De - De(c_idx)*exp(-(D(:,c_idx).^2)/((rb/2)^2));
end

% Compute groups
P = unique(P,'rows');
DP = ai_distmat(X,P,dist,S);
[~,Y] = min(DP,[],2);

end