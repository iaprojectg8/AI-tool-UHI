import pandas as pd
from simpledbf import Dbf5
import numpy as np
from math import *
import json
from PIL import ImageFont
import wandb
import datetime
from wandb.integration.keras import WandbCallback, WandbMetricsLogger, WandbEvalCallback,WandbModelCheckpoint
import warnings
import os 
# This should remove the warning i have but it does not work
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
from keras import layers,models,callbacks,regularizers,optimizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import itertools
from tqdm import tqdm
import scipy as sp
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
import visualkeras
from joblib import dump, load
import shap
import pickle
from utils.variables_path import *
