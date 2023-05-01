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
