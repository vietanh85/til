
# install.packages('arules')
library(arules)
dataset_raw = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

# create sparse matrix
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

# training apriori on data set
rules = apriori(
  data = dataset,
  parameter = list(support = 0.004, confidence = 0.2)
)
summary(rules)

# visualizing
inspect(sort(rules, by = 'lift')[1:10])