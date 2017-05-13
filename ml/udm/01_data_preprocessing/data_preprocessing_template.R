# Importing data set
dataset = read.csv('Data.csv')

# taking care of missing data
dataset$Age = ifelse(
  is.na(dataset$Age), 
  ave(dataset$Age, FUN = function (x) mean(x, na.rm = TRUE)),
  dataset$Age
)
dataset$Salary = ifelse(
  is.na(dataset$Salary),
  mean(dataset$Salary, na.rm = TRUE),
  dataset$Salary
)

# categorical data
dataset$Country = factor(
  dataset$Country,
  levels = c('France', 'Spain', 'Germany'),
  labels = c(1, 2, 3)
)

dataset$Purchased = factor(
  dataset$Purchased,
  levels = c('No', 'Yes'),
  labels = c(0, 1)
)

# spliting data into training set and test set
# install.packages('caTools')
# library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
trainset = subset(dataset, split == TRUE)
testset = subset(dataset, split == FALSE)

# feature scaling
trainset[, 2:3] = scale(trainset[, 2:3])
testset[, 2:3] = scale(testset[, 2:3])





