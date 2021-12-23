from itertools import count
from os import O_NOINHERIT
from typing import Counter
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from pandas.core.frame import DataFrame
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict

fire_data = pd.read_csv('source_csv/FW_Veg_Rem_Combined.csv')

# initial data exploration

print(fire_data.head())
fire_data.drop('Unnamed: 0',axis=1,inplace=True)
print(fire_data.columns)
