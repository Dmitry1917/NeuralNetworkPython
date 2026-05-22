## Neural networks and ML.

# network.py
This is continuation of https://github.com/Dmitry1917/NeuralNetwork, but here I have implemented most of previous functionality in Python and also tried methods, that was hard to do in the old version. Still almost all is MNIST work.
The differences:
- pure python (obviously);
- do not have Adam optimizer and L2-regularization/weights decay difference;
- allow to add and remove layers after network creation;
- first (close to inputs) layers can be skipped during training;
- more flexible learning rate shedules;
- can deform images a bit randomly to inflate training examples, and thus improve result.

# clustering.py
Just tried different clustering methods from sklearn.cluster:
- KMeans;
- DBSCAN;
- HDBSCAN;
- OPTICS;
- Spectral clustering;
- Gaussian mixture;
- Aglomeration clustering (ward and single).

Used them on several examples from table in documentation https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html. The last experiment can be seen in Clustering.png.
Turns out that phrase from there: "the parameters of each of these dataset-algorithm pairs has been tuned to produce good clustering results" was not exaggeration at all. After many experiments it become clear, for example, that with standart parameters (no trying to look better than really is in most cases):
- DBSCAN is the best in not recognizing clusters, there they do not exist at all (1 uniform);
- Spectral, GM and Ward do everything KMeans do, but better (Ward is usually the worst of them);
- OPTICS dont excessively divide arcs because of changing density like DBSCAN or HDBSCAN do (can be seen on ellipse example), but overdivide everything else.

# som.py
Simple implementation of self-organized maps (1D or 2D) and several tests, based on the same samples, that was used in clustering.py plus MNIST database. The results are visualized by showing how many of the labeled samples belong to what neuron, and, in case of 2D data - also there neurons are in space.

