# import data
dataset = read.csv('Mall_Customers.csv')
X = dataset[, 4:5]

# using elbow method to find optimal number of cluster
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Cluster of Clients'))
  
# apply k-means
set.seed(26)
kmeans = kmeans(X, 5, iter.max = 300, nstart = 10)

# vsualizing 
library(cluster)
clusplot(
  X,
  kmeans$cluster,
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