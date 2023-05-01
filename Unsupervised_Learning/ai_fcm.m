function [P,Y] = ai_fcm(X,c,m,tol,dist)
%AI_FCM Fuzzy C-means Clustering
%
%   Inputs:
%   X - Dataset
%   c - Number of centers
%   m - Weighting exponent
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

%% Step 1. Initialize membership matrix
n = size(X,1);
U = rand(c,n);
U = U./sum(U);

%% Iterate
it = 0;
while it < max_iter
    %% Step 2. Calculate fuzzy cluster centers
    P = (U.^m)*X;
    P = P./sum(U.^m,2);
    %% Step 3. Compute cost function
    D = ai_distmat(X,P,dist,S);
    J = 0;
    for i = 1:c
        for j = 1:n
            J = J + (U(i,j)^m)*(D(j,i)^2);
        end
    end
    if abs(J - J_last) <= tol
        break;
    end
    J_last = J;
    %% Step 4. Compute new membership matrix
    for i = 1:c
        for j = 1:n
            s = 0;
            for k = 1:c
                s = s + (D(j,i)/D(j,k))^(2/(m-1));
            end
            U(i,j) = 1 / s;
        end
    end
    % Update iteration counter
    it = it + 1;
end

% Compute groups
[~,Y] = min(D,[],2);

end