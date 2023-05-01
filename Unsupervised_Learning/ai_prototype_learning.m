function [KMR,FCR,HCR] = ai_prototype_learning(X,M,dbg)
%AI_PROTOTYPE_LEARNING Prototype learning
%
%   Inputs:
%   X - Dataset
%   M - Distance-cluster map
%   dbg - Debug flag
%
%   Outputs:
%   KMR - K-means's results
%   FCR - Fuzzy C-means's results
%   HCR - Hierarchical's results

%% Learning 1. K-means

% Results collection
Distance = {};
Silhouettes = [];
DBIndex = [];
Purity = [];
CalinskiHarabasz = [];

for dist = {'euclidean','manhattan','infinity','mahalanobis','cosine'}
    di = dist{1};
    if ~isKey(M,di)
        continue;
    end
    c = M(di);
    for i = 1:100
        % Preprocess
        pd = ai_preprocess(X,true);
        % Run clustering algorithm on working set
        [P,~] = ai_kmeans(pd.wf,c,0.01,di);
        % Obtain labels in test set
        D = ai_distmat(pd.tf,P,di,[]);
        [~,tl] = min(D,[],2);
        % Compute results in test set
        si = evalclusters(pd.tf,tl,'Silhouette').CriterionValues;
        if si == Inf || isnan(si) || ~isreal(si)
            continue;
        end
        db = evalclusters(pd.tf,tl,'DaviesBouldin').CriterionValues;
        if db == Inf || isnan(db) || ~isreal(db)
            continue;
        end
        pu = ai_purity(pd.tf,P,tl,pd.tl);
        if pu == Inf || isnan(pu) || ~isreal(pu)
            continue;
        end
        ca = evalclusters(pd.tf,tl,'CalinskiHarabasz').CriterionValues;
        if ca == Inf || isnan(ca) || ~isreal(ca)
            continue;
        end
        % Collect results
        Distance = [Distance; string(di)];
        Silhouettes = [Silhouettes; si];
        DBIndex = [DBIndex; db];
        Purity = [Purity; pu];
        CalinskiHarabasz = [CalinskiHarabasz; ca];
        % Log
        if dbg == true
            fprintf('K-means (dist. = %s, si = %0.4f, db = %0.4f, ',di,si,db);
            fprintf('pu = %0.4f, ca = %0.4f)\n',pu,ca);
        end
    end
end

% Results table
KMR = table(Distance,Silhouettes,DBIndex,Purity,CalinskiHarabasz);

%% Learning 2. Fuzzy C-means

% Results collection
Distance = {};
Silhouettes = [];
DBIndex = [];
Purity = [];
CalinskiHarabasz = [];

for dist = {'euclidean','manhattan','infinity','mahalanobis','cosine'}
    di = dist{1};
    if ~isKey(M,di)
        continue;
    end
    c = M(di);
    for m = 2:5
        for i = 1:100
            % Preprocess
            pd = ai_preprocess(X,true);
            % Run clustering algorithm on working set
            [P,~] = ai_fcm(pd.wf,c,m,0.01,di);
            % Obtain labels in test set
            D = ai_distmat(pd.tf,P,di,[]);
            [~,tl] = min(D,[],2);
            % Compute results in test set
            si = evalclusters(pd.tf,tl,'Silhouette').CriterionValues;
            if si == Inf || isnan(si) || ~isreal(si)
                continue;
            end
            db = evalclusters(pd.tf,tl,'DaviesBouldin').CriterionValues;
            if db == Inf || isnan(db) || ~isreal(db)
                continue;
            end
            pu = ai_purity(pd.tf,P,tl,pd.tl);
            if pu == Inf || isnan(pu) || ~isreal(pu)
                continue;
            end
            ca = evalclusters(pd.tf,tl,'CalinskiHarabasz').CriterionValues;
            if ca == Inf || isnan(ca) || ~isreal(ca)
                continue;
            end
            % Collect results
            Distance = [Distance; string(di)];
            Silhouettes = [Silhouettes; si];
            DBIndex = [DBIndex; db];
            Purity = [Purity; pu];
            CalinskiHarabasz = [CalinskiHarabasz; ca];
            % Log
            if dbg == true
                fprintf('Fuzzy C-means (dist. = %s, si = %0.4f, db = %0.4f, ',di,si,db);
                fprintf('pu = %0.4f, ca = %0.4f)\n',pu,ca);
            end
        end
    end
end

% Results table
FCR = table(Distance,Silhouettes,DBIndex,Purity,CalinskiHarabasz);

%% Learning 3. Hierarchical clustering

% Results collection
Distance = {};
Silhouettes = [];
DBIndex = [];
Purity = [];
CalinskiHarabasz = [];

for dist = {'euclidean','manhattan','infinity','mahalanobis','cosine'}
    di = dist{1};
    if ~isKey(M,di)
        continue;
    end
    c = M(di);
    for i = 1:100
        % Preprocess
        pd = ai_preprocess(X,true);
        % Run clustering algorithm on working set
        [P,~] = ai_hierarchical(pd.wf,c,di);
        % Obtain labels in test set
        D = ai_distmat(pd.tf,P,di,[]);
        [~,tl] = min(D,[],2);
        % Compute results in test set
        si = evalclusters(pd.tf,tl,'Silhouette').CriterionValues;
        if si == Inf || isnan(si) || ~isreal(si)
            continue;
        end
        db = evalclusters(pd.tf,tl,'DaviesBouldin').CriterionValues;
        if db == Inf || isnan(db) || ~isreal(db)
            continue;
        end
        pu = ai_purity(pd.tf,P,tl,pd.tl);
        if pu == Inf || isnan(pu) || ~isreal(pu)
            continue;
        end
        ca = evalclusters(pd.tf,tl,'CalinskiHarabasz').CriterionValues;
        if ca == Inf || isnan(ca) || ~isreal(ca)
            continue;
        end
        % Collect results
        Distance = [Distance; string(di)];
        Silhouettes = [Silhouettes; si];
        DBIndex = [DBIndex; db];
        Purity = [Purity; pu];
        CalinskiHarabasz = [CalinskiHarabasz; ca];
        % Log
        if dbg == true
            fprintf('Hierarchical (dist. = %s, si = %0.4f, db = %0.4f, ',di,si,db);
            fprintf('pu = %0.4f, ca = %0.4f)\n',pu,ca);
        end
    end
end

% Results table
HCR = table(Distance,Silhouettes,DBIndex,Purity,CalinskiHarabasz);

end