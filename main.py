import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
#from sklearn import tree

data = pd.read_csv("train.csv")

print(data.head())

print(data.shape)
print(data.columns)

# Scelta delle variabili di input e output
y = data["price_range"]
print("y", y)
X = data.drop("price_range", axis = 1)
print("X", X)

# Suddivisione dei dati per le fasi di training e validazione
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.8, random_state = 7)

print(train_X.shape)

print(val_X.shape)

# Creazione del classificatiore e adattamento ai dati di training
model = RandomForestClassifier(random_state = 7, n_estimators = 100)
model.fit(train_X, train_y)

# Predizione delle classi per i dati di validazione

"""

La classe prevista di un campione di input è un voto degli alberi nella foresta, 
ponderato in base alle loro stime di probabilità. 
Cioè, la classe prevista è quella con la stima della probabilità media più alta 
tra gli alberi.

"""

pred_y = model.predict(val_X)

# Calcolo dell"accuratezza del modello
accuracy = metrics.accuracy_score(val_y, pred_y)
print("Accuracy: ", accuracy)

# Calcolo della matrice di confusione
confusion = metrics.confusion_matrix(val_y, pred_y)
print("Confusion matrix:\n{confusion}")


# Normalizzazione della matrice di confusione
print("\nNormalized confusion matrix:")
for row in confusion:
   print(row / row.sum())

print("\n")

"""

Le probabilità delle classi previste di un campione di input vengono calcolate come 
la media delle probabilità delle classi previste degli alberi nella foresta. 
La probabilità di classe di un singolo albero è la frazione di campioni della 
stessa classe in una foglia.

"""

probs = model.predict_proba(val_X)
print(probs)
