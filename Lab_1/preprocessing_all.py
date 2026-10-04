import numpy as np
from sklearn import preprocessing

# Исходные данные
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

# 1.1 Бинаризация
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# 1.2 Исключение среднего
# Удаляем среднее, чтобы каждый признак был центрирован на 0
data_scaled = preprocessing.scale(input_data)
print("\nMean removed (Mean =", data_scaled.mean(axis=0), ")")
print("Std deviation =", data_scaled.std(axis=0))

# 1.3 Масштабирование 
# Приводим значения в диапазон от 0 до 1
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# 1.4 Нормализация
# L1 - сумма абсолютных значений равна 1
# L2 - сумма квадратов равна 1
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)