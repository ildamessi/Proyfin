import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv('dataset_algebra.csv')

# Limpieza
df['Tiempo_Segundos'] = df['Tiempo_Segundos'].apply(lambda x: x if x >= 0 else None)
df[['Tiempo_Segundos']] = SimpleImputer(strategy='median').fit_transform(df[['Tiempo_Segundos']])
df[['Dificultad_Percibida']] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Dificultad_Percibida']])

# Codificación de variables categóricas
le_dict = {}
for col in ['Tipo_Problema', 'Dificultad_Percibida', 'Resuelto_Correctamente']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Definir X e y
X = df.drop(['ID_Estudiante', 'Resuelto_Correctamente'], axis=1)
y = df['Resuelto_Correctamente']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("REPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Importancia de características
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nIMPORTANCIA DE VARIABLES:")
print(importances)

