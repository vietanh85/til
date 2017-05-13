# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv')
transations = []
for i in range(0, 7501):
    transations.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
from apyori import apriori
rules = apriori(transations, min_support=0.004, min_confidence=0.2, min_lift=3, min_length=2)
result = list(rules)