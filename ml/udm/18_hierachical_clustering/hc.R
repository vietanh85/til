# Hierarchical Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# using dentogram to find optimal cluster number
dentogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dentogram)

# fitting model
y_hc = cutree(dentogram, k = 5)

# vsualizing 
library(cluster)
clusplot(
  X,
  y_hc,
  lines = 0,
  shade = TRUE,
  color = TRUE,
  labels = 2,
  plotchar = FALSE,
  span = TRUE,
  main = 'Cluster of client',
  xlab = 'Income',
  ylab = 'Score'
)