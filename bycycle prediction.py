import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

data_path = 'data/hour.csv'
rides = pd.read_csv(data_path)
print(rides.head())
counts = rides['cnt'][:50]
