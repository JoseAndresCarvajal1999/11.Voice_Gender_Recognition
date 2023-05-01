function D = ai_distmat(X,R,dist,S)
%AI_DISTMAT Distance matrix
%
%   Inputs:
%   X - Points
%   R - Reference points
%   dist - Distance type
%   S - Covariance matrix
%
%   Outputs:
%   D - Distance matrix

% Dimensions
n = size(X,1);
c = size(R,1);

% Initialization
D = zeros(n,c);
if isempty(S)
    S = cov(X);
end

% Calculate distances to reference points
for i = 1:c
    r = R(i,:);
    for j = 1:n
        x = X(j,:);
        switch dist
            case 'manhattan'
                D(j,i) = ai_manhattan(x-r);
            case 'euclidean'
                D(j,i) = ai_euclidean(x-r);
            case 'infinity'
                D(j,i) = ai_infinity(x-r);
            case 'mahalanobis'
                D(j,i) = ai_mahalanobis(x,r,S);
            case 'cosine'
                D(j,i) = ai_cosine(x,r);
            otherwise
                error('Unsupported distance type');
        end
    end
end

end