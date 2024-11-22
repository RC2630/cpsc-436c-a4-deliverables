from sklearn import datasets # type: ignore
import pickle
X, y = datasets.make_regression(100, 1, noise=5, bias=0)
pickle.dump([X,y], open('./train.pickle', 'wb'))
from sklearn.linear_model import LinearRegression # type: ignore
[XX, yy] = pickle.load(open('./train.pickle', 'rb'))
model = LinearRegression()
model.fit(XX,yy)
print(model.predict([[0],[1],[2],[3]]))
p = pickle.dumps(model)
pickle.dump(model, open('./model.pickle', 'wb'))
loaded_model = pickle.load(open('./model.pickle', 'rb'))
print(loaded_model.predict([[0],[1],[2],[3]]))