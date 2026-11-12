from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

X,y=make_classification(n_samples=200,n_features=2,n_informative=2,
                        n_redundant=0,n_clusters_per_class=1,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
print("Precision:",round(precision_score(y_test,y_pred)*100,2),"%")
print("Recall   :",round(recall_score(y_test,y_pred)*100,2),"%")
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))