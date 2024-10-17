import numpy as np
from RS import RS
from data import loadtxt, to_vector, fast_loadtxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

population_size = 2000
dimension = 10
cluster_number = 10

# population, probability = loadtxt(population_size, dimension, cluster_number)
population, probability = fast_loadtxt(population_size, dimension)
population_vector = to_vector(population)
X = population
y = probability.reshape((-1, population_size, 1))
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# recommender_system = RS(X_train, y_train, 'model_test.h5')
recommender_system = RS(X_train, y_train)
recommender_system.fit(epochs=100, batch_size=50)
recommender_system.save('model_test1.h5')
# 测试预测
prediction = recommender_system.predict(X_test)
y_prediction = prediction.reshape((-1, population_size))
y_test = y_test.reshape((-1, population_size))
while(True):
    index = int(np.random.rand() * len(y_test))
    plt.plot(y_test[index], label='True')
    plt.plot(y_prediction[index], label='Predicted')
    plt.xlabel('individual')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()
    input()