function pu = ai_purity(X,P,Y,Yr)
%AI_PURITY Purity Index
%
%   Inputs:
%   X - Dataset
%   P - Prototypes / Centers
%   Y - Group assignments
%   Yr - Real group assignments
%
%   Outputs:
%   pu - Purity

% Number of clusters
n = size(P,1);

% Number of points
m = size(X,1);

% Summation
s = 0;
for i = 1:n
    idxK = (Y==i);
    yRk = Yr(idxK,:);
    [~,fr] = mode(yRk);
    s = s + fr;
end

% Purity Index
pu = s / m;

end