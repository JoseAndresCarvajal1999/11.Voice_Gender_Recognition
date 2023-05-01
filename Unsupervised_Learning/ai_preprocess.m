function out = ai_preprocess(X,labeled)
%AI_PREPROCESS Preprocess dataset
%
%   Inputs:
%   X - Dataset
%   labeled - Labeled flag
%
%   Outputs:
%   out - Preprocessing results

% Output structure
out = struct;

% Extract labels
if labeled == true
    y = X(:,end);
    X = X(:,1:end-1);
    out.Y = y;
end

% Scale dimensions to unit hypercube
X = (X-min(X))./(max(X)-min(X));
out.X = X;

% Partition dataset in validation and rest
val_pt = cvpartition(size(X,1),'HoldOut',0.2);
idx_val = test(val_pt);
idx_rest = training(val_pt);
out.vf = X(idx_val,:);
X = X(idx_rest,:);

% Partition remaining dataset in work and test
test_pt = cvpartition(size(X,1),'HoldOut',0.2);
idx_test = test(test_pt);
idx_work = training(test_pt);
out.tf = X(idx_test,:);
out.wf = X(idx_work,:);

% Fill labels
if labeled == true
    out.vl = y(idx_val,:);
    out.tl = y(idx_test,:);
    out.wl = y(idx_work,:);
end

end