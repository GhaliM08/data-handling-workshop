import pandas as pd

titanic = pd.read_csv('titanic.csv')

# ---- explore + detect
print(titanic.shape)
print(titanic.isna().sum()[titanic.isna().sum() > 0])
print('duplicated rows:', titanic.duplicated().sum())
print(titanic['embarked'].value_counts(dropna=False))
print(titanic['fare'].eq(0).sum(), 'fare == 0 rows')

# ---- clean (work on the same dataframe, keep flags)
titanic['age_imputed'] = titanic['age'].isna()
titanic['age'] = titanic['age'].fillna(titanic.groupby(['pclass', 'sex'])['age'].transform('median'))

titanic['embarked_imputed'] = titanic['embarked'].isna()
titanic.loc[titanic['embarked'].isna(), ['embarked', 'embark_town']] = ['C', 'Cherbourg']

titanic['deck_known'] = titanic['deck'].notna()
titanic['deck'] = titanic['deck'].fillna('Unknown')

titanic['fare_zero'] = titanic['fare'].eq(0)

# ---- check
print(titanic.isna().sum().sum(), 'missing values left')
print(round(titanic['age'].mean(), 2), 'mean age')
print(titanic[['age_imputed', 'embarked_imputed', 'deck_known', 'fare_zero']].sum().to_dict())
