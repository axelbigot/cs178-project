from ucimlrepo import fetch_ucirepo


# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets

if __name__ == '__main__':
    pass
