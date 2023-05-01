function d = ai_cosine(x,y)
%AI_COSINE Cosine distance
%
%   Inputs:
%   x - First point
%   y - Second point
%
%   Outputs:
%   d - Distance

d = dot(x,y)/(ai_euclidean(x)*ai_euclidean(y));

end