import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection 
from utilities import visualize_classifier

# Входной файл
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Создаем классификатор
classifier = GaussianNB()

# Обучаем
classifier.fit(X, y)

# Визуализируем
visualize_classifier(classifier, X, y)

# Оценка качества 
num_folds = 3
accuracy_values = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")