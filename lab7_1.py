from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Звіт про класифікацію:")
print(classification_report(y_test, predictions))

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Прогноз моделі')
plt.ylabel('Реальний вид')
plt.title('Матриця помилок (Confusion Matrix)')
plt.show()

plt.figure(figsize=(8,6))
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Важливість ознак для моделі')
plt.show()