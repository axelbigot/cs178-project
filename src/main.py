from ucimlrepo import fetch_ucirepo


# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets

# Normalizing some data. Some income fields have trailing period,
# some don't. Remove trailing period for consistency.
y = y.applymap(lambda v: v.rstrip('.'))

if __name__ == '__main__':
    pass
