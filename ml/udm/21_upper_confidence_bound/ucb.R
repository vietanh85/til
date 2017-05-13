
dataset = read.csv('Ads_CTR_Optimisation.csv')

# UCB
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
number_of_selections = integer(d)
sum_of_rewards = integer(d)
for (n in 1:N) {
  ad = 0
  max_ucb = 0
  for (i in 1:d) {
    if (number_of_selections[i] > 0) {
      avg_reward = sum_of_rewards[i] / number_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / number_of_selections[i])
      ucb = avg_reward + delta_i
    } else {
      ucb = 1e400
    }
    if (ucb > max_ucb) {
      max_ucb = ucb
      ad = i
    }
  }
  
  ads_selected = append(ads_selected, ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = dataset[n, ad]
  sum_of_rewards[ad] = sum_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# random selection
# N = 10000
# d = 10
# ads_selected = integer(0)
# total_reward = 0
# for (n in 1:N) {
#   ad = sample(1:10, 1)
#   ads_selected = append(ads_selected, ad)
#   reward = dataset[n, ad]
#   total_reward = total_reward + reward
# }

# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')