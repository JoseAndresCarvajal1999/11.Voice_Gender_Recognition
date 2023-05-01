# 11.Voice_Gender_Recognition

For this project, an unsupervised and supervised learning process was carried out as follows:

- Unsupervised learning: 

  - This work applies five different clustering techniques (Mountain Clustering, Subtractive Clustering, K-means,
  Fuzzy C-means, and Hierarchical Clustering) with several norms
  (Euclidean, Manhattan, Infinity, and Mahalanobis) to a Voice
  Gender dataset that was built to distinguish a voice as male or female, based upon their speech’s acoustic properties. Case in point,
  the workflow builds two additional spaces, a high-dimensional
  space with polynomial features and a 2-dimensional space using
  Barnes Hut t-SNE algorithm, to assess which exploration and
  learning setup renders the most suitable clustering properties
  as measured by several internal validation indexes. Results show
  that clustering results are highly dependant on the elected spacenorm combination. In addition, evidence shows that the proposed
  polynomial transformation does not furnish the expected benefits.
  Contrastingly, the embedding procedure seems to induce new
  partitions in the dataset that might not be related to the ground
  truth
  
 
- Supervised Learning 
    - This work applies five different supervised learning
  techniques (Decision Tree, Linear SVM, Polynomial SVM, RBF
  SVM, and Multilayer Perceptron) to a Voice Gender dataset that
  was built to distinguish a voice as male or female, based upon
  their speech’s audile properties. Case in point, the workflow
  operates within two spaces, a high-dimensional space of 20
  acoustic features and a 2-dimensional space using Barnes Hut
  t-SNE algorithm, to assess which learner renders the most
  suitable performance metrics in each scenario. Results show
  that, in the high-dimensional space, the Decision Tree and the
  L = [20, 9, 9, 10, 2] with η = 0.9 MLP setup tied in the first
  place as the most competent learning machines with accuracy
  figures above the 0.98 mark. In addition, no sufficient evidence
  was found to point out that the complexity of the architecture has
  a strong relationship with the obtained metrics in that scenario.
  Contrastingly, the embedding procedure furnished no meaningful
  benefits within this study, as neither trained models performed
  decently in that context.
