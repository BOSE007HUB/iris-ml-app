import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import pickle 
df = pd.read_csv("iris.csv") 
X = df.drop("species", axis=1) 
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
model = KNeighborsClassifier(n_neighbors=3) 
model.fit(X_train, y_train) 
# Save the model 
pickle.dump(model, open("model.pkl", "wb"))
