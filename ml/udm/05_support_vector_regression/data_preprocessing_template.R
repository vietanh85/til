# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# dataset = scale(dataset)

# fitting linear svr to dataser
# install.packages('e1071')
library(e1071)
svr = svm(
  formula = Salary ~.,
  data = dataset,
  type = 'eps-regression'
)

lin_reg = lm(
  formula = Salary ~ Level,
  data = dataset
)

# fitting svr regression to dataser


# visualizing 
library(ggplot2)
ggplot() +
  geom_point(
    aes(
      x = dataset$Level,
      y = dataset$Salary
    ),
    colour='red'
  ) +
  geom_line(
    aes(
      x = dataset$Level,
      y = predict(lin_reg, newdata = dataset)
    ),
    colour='blue'
  ) +
  geom_line(
    aes(
      x = dataset$Level,
      y = predict(svr, newdata = dataset)
    ),
    colour='green'
  )

# predicting
y_pred1 = predict(lin_reg, newdata = data.frame(Level = 6.5))
# y_pred2 = predict(lin_reg2, newdata = data.frame(Level = 6.5, 
#                                                  Level2 = 6.5^2, 
#                                                  Level3 = 6.5^3, 
#                                                  Level4 = 6.5^4))
