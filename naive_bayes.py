from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,precision_score,recall_score

data=load_iris()
X,y=data.data,data.target

Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,
                              random_state=42,stratify=y)

model=GaussianNB().fit(Xt,yt)
p=model.predict(Xv)

print("Accuracy :",round(accuracy_score(yv,p)*100,2),"%")
print("Precision:",round(precision_score(yv,p,average="weighted")*100,2),"%")
print("Recall   :",round(recall_score(yv,p,average="weighted")*100,2),"%")