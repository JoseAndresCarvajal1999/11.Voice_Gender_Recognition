function [P,Y] = ai_kmeans(X,c,tol,dist)
%AI_KMEANS K-means Clustering
%
%   Inputs:
%   X - Dataset
%   c - Number of centers
%   tol - Tolerance
%   dist - Distance type
%
%   Outputs:
%   P - Prototypes / Centers
%   Y - Groups

% Last cost
J_last = Inf;

% Maximum number of iterations
max_iter = 10000;

% Covariance matrix
S = cov(X);

%% Step 1. Initialize cluster centers
n = size(X,1);
idx = randperm(n,c);
P = X(idx,:);

%% Iterate
it = 0;
while it < max_iter
    %% Step 2. Determine membership matrix
    U = false(c,n);
    D = ai_distmat(X,P,dist,S);
    for i = 1:c
        for j = 1:n
            s = 0;
            for k = 1:c
                if k ~= i
                    if D(j,i)^2 <= D(j,k)^2
                        s = s + 1;
                    else
                        break;
                    end
                end
            end
            if s == c-1
                U(i,j) = true;
            end
        end
    end
    %% Step 3. Compute cost function
    J = 0;
    for i = 1:c
        G = U(i,:);
        J = J + sum(D(G,i).^2);
    end
    if abs(J - J_last) <= tol
        break;
    end
    J_last = J;
    %% Step 4. Update cluster centers
    for i = 1:c
        idx = U(i,:);
        G = X(idx,:);
        P(i,:) = mean(G);
    end
    % Update iteration counter
    it = it + 1;
end

% Compute groups
[~,Y] = min(D,[],2);

end