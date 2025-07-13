#####cross validation 
##classification 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()

# 5-Fold CV
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())



##regression 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
model = LinearRegression()

scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
print("MSE (per fold):", -scores)
