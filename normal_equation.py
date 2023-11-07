from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[2, 3], [4, 5], [1, 7], [5, 2], [8, 9], [10, 1], [6, 4], [3, 7], [9, 3], [7, 5]])
y = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'])

OvA_model={}
clasess=np.unique(y)
for c in clasess:
    y_binary=np.where(y==c,1,0)

    model=LogisticRegression()
    model.fit(X,y_binary)

    OvA_model[c]=model
train_accuracy= {}
for c, model in OvA_model.items():
    train_accuracy[c]= model.score(X,np.where(y==c,1,0))
print(" acurratecy of training:")
print(train_accuracy)

new_point=np.array([[6,6]])
predictor={}
for c, model in OvA_model.items():
    predictor[c] = model.predict(new_point)
print(" predict new point:")
print(predictor)