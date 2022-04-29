#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sklearn as sk
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# In[79]:


def rec_per(sample, item_corr_mat_mem_based_per, game_names = game_names):
    idx_sample = game_names.index(sample)
    top_10_sim_games_mem_based_per = item_corr_mat_mem_based_per[idx_sample].argsort()[::-1][:11]
    top_10_sim_games = [game_names[x] for x in top_10_sim_games_mem_based_per]
    return top_10_sim_games

def rec_cs(sample, item_corr_mat_mem_based, game_names = game_names):
    idx_sample = game_names.index(sample)
    top_10_sim_games_mem_based = item_corr_mat_mem_based[idx_sample].argsort()[::-1][:11]
    top_10_sim_games = [game_names[x] for x in top_10_sim_games_mem_based]
    return top_10_sim_games

def rec_mf(sample, item_corr_mat, game_names = game_names):
    idx_sample = game_names.index(sample)
    top_10_sim_games = item_corr_mat[idx_sample].argsort()[::-1][:11]
    top_10_sim_games_names = [game_names[x] for x in top_10_sim_games]
    return top_10_sim_games_names

def game_title_rec(game_title, item_corr_mat_mem_based_per, item_corr_mat_mem_based, item_corr_mat):
    sample_per = rec_per(game_title, item_corr_mat_mem_based_per)
    sample_cs = rec_cs(game_title, item_corr_mat_mem_based)
    sample_mf = rec_mf(game_title, item_corr_mat)
    
    print("The original game and the top 10 most recommended games (based on Pearson correlation):\n")
    print(sample_per)
    print("\n")
    
    print("The original game and the top 10 most recommended games (based on Cosine correlation):\n")
    print(sample_cs)
    print("\n")
    
    print("The original game and the top 10 most recommended games (based on Matrix factorization correlation):\n")
    print(sample_mf)
    print("\n")


# In[71]:


filename = "steam-200k.csv"
colnames = ['User ID', 'Name of the Game', 'Behavior Type', 'No. of Hours', '0']
df = pd.read_csv(filename,  error_bad_lines=False, encoding='utf-8', names=colnames)
df = df.drop(['0'], axis=1)
df = df[df['Behavior Type'] != 'purchase']
df_behav = df.drop(['Behavior Type'], axis=1)

matrix = df.pivot_table(columns='Name of the Game', index='User ID', values='No. of Hours', fill_value=0)
game_names = list(matrix.columns)

non_zeros = 0
for col in matrix.columns:
    column = matrix[col]
    non_zeros += (column!=0).sum()
total_matrix_entries = matrix.shape[0]*matrix.shape[1]
sparsity = (non_zeros/total_matrix_entries)*100
print("The user-item matrix has this percent of non-zero values :" + str(sparsity))


# In[72]:


# Item correlation matrix using Cosine Similarity
item_corr_mat_mem_based = cosine_similarity(matrix.transpose())

# Item correlation matrix using Pearson Correlation
item_corr_mat_mem_based_per = np.corrcoef(matrix.transpose())

# item correaltion matrix using Matrix factorization
epsilon = 1e-9
n_latent_factors = 5

# calculate item latent matrix
item_svd = TruncatedSVD(n_components = n_latent_factors)
item_features = item_svd.fit_transform(matrix.transpose()) + epsilon

# compute similarity
item_corr_mat = cosine_similarity(item_features)


# In[78]:


# Sample List for 5 example games

samples = ['Prototype', 'Team Fortress 2', 'Portal', 'Spore', 'Dota 2']

for x in samples:
    sample_per = rec_per(x, item_corr_mat_mem_based_per)
    sample_cs = rec_cs(x, item_corr_mat_mem_based)
    sample_mf = rec_mf(x, item_corr_mat)
    
    print("The original game and the top 10 most recommended games (based on Pearson correlation):\n")
    print(sample_per)
    print("\n")
    
    print("The original game and the top 10 most recommended games (based on Cosine correlation):\n")
    print(sample_cs)
    print("\n")
    
    print("The original game and the top 10 most recommended games (based on Matrix factorization correlation):\n")
    print(sample_mf)
    print("\n")


# In[83]:


### USER INPUT CODE

print("Welcome to the Game Recommendation System !!!\n")

game_title = input("Please enter the name of your game (Make sure to enter the title from unique game list in the github repo):")

print("\n")
game_title_rec(game_title, item_corr_mat_mem_based_per, item_corr_mat_mem_based, item_corr_mat)

