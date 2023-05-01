function d = ai_mahalanobis(x,y,S)
%AI_MAHALANOBIS Mahalanobis distance
%
%   Inputs:
%   x - First point
%   y - Second point
%   S - Covariance matrix
%
%   Outputs:
%   d - Distance

d = sqrt((x-y)*inv(S)*(x-y)');

end