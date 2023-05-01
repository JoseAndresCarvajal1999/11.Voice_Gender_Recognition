function [MR,SR] = ai_prototype_exploration(X,mnt,dbg)
%AI_PROTOTYPE_EXPLORATION Prototype exploration
%
%   Inputs:
%   X - Dataset
%   mnt - Mountain flag
%   dbg - Debug flag
%
%   Outputs:
%   MR - Mountain's results
%   SR - Subtractive's results

% Preprocess
pd = ai_preprocess(X,true);

%% Exploration 1. Mountain Clustering
MR = [];

if mnt == true
    
    % Results collection
    Distance = {};
    Sigma = [];
    Beta = [];
    Clusters = [];
    Silhouettes = [];
    DBIndex = [];
    Purity = [];
    CalinskiHarabasz = [];
    
    % Brute-force exploration
    for dist = {'euclidean','manhattan','infinity','mahalanobis','cosine'}
        di = dist{1};
        for sigma = 0.1:0.1:1
            for beta = 0.1:0.1:1
                % Run clustering algorithm
                [P,Y] = ai_mountain(pd.X,0.1,sigma,beta,0.01,di);
                % Compute results
                si = evalclusters(pd.X,Y,'Silhouette').CriterionValues;
                if si == Inf || isnan(si) || ~isreal(si)
                    continue;
                end
                db = evalclusters(pd.X,Y,'DaviesBouldin').CriterionValues;
                if db == Inf || isnan(db) || ~isreal(db)
                    continue;
                end
                pu = ai_purity(pd.X,P,Y,pd.Y);
                if pu == Inf || isnan(pu) || ~isreal(pu)
                    continue;
                end
                ca = evalclusters(pd.X,Y,'CalinskiHarabasz').CriterionValues;
                if ca == Inf || isnan(ca) || ~isreal(ca)
                    continue;
                end
                k = size(P,1);
                % Collect results
                Distance = [Distance; string(di)];
                Sigma = [Sigma; sigma];
                Beta = [Beta; beta];
                Clusters = [Clusters; k];
                Silhouettes = [Silhouettes; si];
                DBIndex = [DBIndex; db];
                Purity = [Purity; pu];
                CalinskiHarabasz = [CalinskiHarabasz; ca];
                % Log
                if dbg == true
                    fprintf('Mountain (dist. = %s, sigma = %0.2f, ',di,sigma);
                    fprintf('beta = %0.2f, k = %d, ',beta,k);
                    fprintf('si = %0.4f, db = %0.4f, ',si,db);
                    fprintf('pu = %0.4f, ca = %0.4f)\n',pu,ca);
                end
            end
        end
    end
    
    % Results table
    MR = table(Distance,Sigma,Beta,Clusters,Silhouettes,DBIndex,Purity,CalinskiHarabasz);
    
end

%% Exploration 2. Subtractive Clustering

% Results collection
Distance = {};
Ra = [];
Rb = [];
Clusters = [];
Silhouettes = [];
DBIndex = [];
Purity = [];
CalinskiHarabasz = [];

% Brute-force exploration
for dist = {'euclidean','manhattan','infinity','mahalanobis','cosine'}
    di = dist{1};
    for ra = 0.1:0.1:1
        rb = 1.5*ra;
        % Run clustering algorithm
        [P,Y] = ai_substractive(pd.X,ra,rb,0.01,di);
        % Compute results
        si = evalclusters(pd.X,Y,'Silhouette').CriterionValues;
        if si == Inf || isnan(si) || ~isreal(si)
            continue;
        end
        db = evalclusters(pd.X,Y,'DaviesBouldin').CriterionValues;
        if db == Inf || isnan(db) || ~isreal(db)
            continue;
        end
        pu = ai_purity(pd.X,P,Y,pd.Y);
        if pu == Inf || isnan(pu) || ~isreal(pu)
            continue;
        end
        ca = evalclusters(pd.X,Y,'CalinskiHarabasz').CriterionValues;
        if ca == Inf || isnan(ca) || ~isreal(ca)
            continue;
        end
        k = size(P,1);
        % Collect results
        Distance = [Distance; string(di)];
        Ra = [Ra; ra];
        Rb = [Rb; rb];
        Clusters = [Clusters; k];
        Silhouettes = [Silhouettes; si];
        DBIndex = [DBIndex; db];
        Purity = [Purity; pu];
        CalinskiHarabasz = [CalinskiHarabasz; ca];
        % Log
        if dbg == true
            fprintf('Subtractive (dist. = %s, ra = %0.2f, ',di,ra);
            fprintf('rb = %0.2f, k = %d, ',rb,k);
            fprintf('si = %0.4f, db = %0.4f, ',si,db);
            fprintf('pu = %0.4f, ca = %0.4f)\n',pu,ca);
        end
    end
end

% Results table
SR = table(Distance,Ra,Rb,Clusters,Silhouettes,DBIndex,Purity,CalinskiHarabasz);

end