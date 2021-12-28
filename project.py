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
# dropping duplicate index column
fire_data = fire_data.drop('Unnamed: 0',axis='columns')
fire_data = fire_data.drop('Unnamed: 0.1',axis='columns')
# print(fire_data.columns)

print(fire_data['stat_cause_descr'].unique())
# 13 unique fire causes 
# 'Missing/Undefined' 'Arson' 'Debris Burning' 'Miscellaneous' 'Campfire'
# 'Fireworks' 'Children' 'Lightning' 'Equipment Use' 'Smoking' 'Railroad'
# 'Structure' 'Powerline'
print(fire_data['stat_cause_descr'].value_counts())




