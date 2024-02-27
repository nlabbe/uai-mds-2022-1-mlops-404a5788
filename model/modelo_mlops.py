#Importar paquetes
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

#Importar dataset
url_data = "./Iris.csv"
iris = pd.read_csv(url_data)
y = iris['Species'].copy()
X = iris.drop('Species',index=None,axis=1).copy()

X.drop(['Id'],axis=1,inplace=True)

# Separar los set de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Entrenamiento
tree_model = DecisionTreeClassifier()

params = {'max_depth': [2,4,6,8,10,12],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}

opt_tree_model = GridSearchCV(tree_model,param_grid=params)
opt_tree_model.fit(X_train,y_train)
pred_values_otree = opt_tree_model.predict(X_test)

print(opt_tree_model.best_params_)
print(classification_report(y_test,pred_values_otree))
print(confusion_matrix(y_test,pred_values_otree))
# Se guardan los metadatos del mejor modelo
metadatos = pd.DataFrame(opt_tree_model.best_params_,index=range(1))
joblib.dump(opt_tree_model, './model.pkl')
print(metadatos)
print(classification_report(y_test,pred_values_otree))
print(confusion_matrix(y_test,pred_values_otree))

