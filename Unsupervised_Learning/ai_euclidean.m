function d = ai_euclidean(x)
%AI_EUCLIDEAN Euclidean norm
%
%   Inputs:
%   x - Point
%
%   Outputs:
%   d - Norm

d = sum(x.^2)^(1/2);

end