import numpy as np

def loadtxt(population_size, dimension, cluster_number):
    population = np.loadtxt('population.txt')
    population = population.reshape(-1, population_size, dimension)
    clusters = np.loadtxt('cluster.txt')
    clusters = clusters.reshape(-1, cluster_number).astype(np.int32)
    fitness = np.loadtxt('fitness.txt')
    fitness = fitness.reshape(-1, population_size)
    probability = np.zeros((len(population), population_size))
    for i in range(len(population)): # 根据fitness组内排序
        cnt = 0
        for j in range(cluster_number):
            population[i][cnt:cnt+clusters[i][j]], fitness[i][cnt:cnt+clusters[i][j]] = merge_sort(population[i][cnt:cnt+clusters[i][j]], fitness[i][cnt:cnt+clusters[i][j]])
            f = fitness[i][cnt:cnt+clusters[i][j]]
            f_min = np.min(f)
            probability[i][cnt:cnt+clusters[i][j]] = 1 - (f - f_min) / f
            cnt += clusters[i][j]
    for i in range(len(population)):
        population[i], fitness[i], probability[i] = merge_sort_all(population[i], fitness[i], probability[i])
    with open('fast_population.txt', 'a') as f1, open('fast_probability.txt', 'a') as f2:
        for i in range(len(population)):
            np.savetxt(f1, population[i])
        np.savetxt(f2, probability)
    return population, probability

def to_vector(population): # 转成由重心指向点的向量
    center = np.mean(population, axis=0)
    vector = population - np.expand_dims(center, axis=0)
    vector = vector / (np.expand_dims(np.std(population, axis=0), axis=0) + 10e-6)
    return vector

def merge(array1, array2, fitness1, fitness2):
    population_result = np.zeros((len(array1)+len(array2), array1.shape[1]))
    fitness_result = np.zeros(len(array1)+len(array2))
    i1, i2 = 0, 0
    f1 = fitness1[0]
    f2 = fitness2[0]
    for i in range(len(array1) + len(array2)):
        if f1 >= f2:
            population_result[i] = array1[i1]
            fitness_result[i] = fitness1[i1]
            i1 += 1
            if i1 == len(array1):
                population_result[i+1:] = array2[i2:]
                fitness_result[i+1:] = fitness2[i2:]
                break
            else:
                f1 = fitness1[i1]
        else:
            population_result[i] = array2[i2]
            fitness_result[i] = fitness2[i2]
            i2 += 1
            if i2 == len(array2):
                population_result[i+1:] = array1[i1:]
                fitness_result[i+1:] = fitness1[i1:]
                break
            else:
                f2 = fitness2[i2]
    return population_result, fitness_result

def merge_sort(array, fitness):
    if len(array) <= 1:
        return array, fitness
    else:
        mid = int(len(array) / 2)
        array1, fitness[:mid] = merge_sort(array[:mid], fitness[:mid])
        array2, fitness[mid:] = merge_sort(array[mid:], fitness[mid:])
        return merge(array1, array2, fitness[:mid], fitness[mid:])
    
def merge_all(array1, array2, fitness1, fitness2, probability1, probability2):
    population_result = np.zeros((len(array1)+len(array2), array1.shape[1]))
    fitness_result = np.zeros(len(array1)+len(array2))
    probability_result = np.zeros(len(array1)+len(array2))
    i1, i2 = 0, 0
    f1 = fitness1[0]
    f2 = fitness2[0]
    for i in range(len(array1) + len(array2)):
        if f1 >= f2:
            population_result[i] = array1[i1]
            fitness_result[i] = fitness1[i1]
            probability_result[i] = probability1[i1]
            i1 += 1
            if i1 == len(array1):
                population_result[i+1:] = array2[i2:]
                fitness_result[i+1:] = fitness2[i2:]
                probability_result[i+1:] = probability2[i2:]
                break
            else:
                f1 = fitness1[i1]
        else:
            population_result[i] = array2[i2]
            fitness_result[i] = fitness2[i2]
            probability_result[i] = probability2[i2]
            i2 += 1
            if i2 == len(array2):
                population_result[i+1:] = array1[i1:]
                fitness_result[i+1:] = fitness1[i1:]
                probability_result[i+1:] = probability1[i1:]
                break
            else:
                f2 = fitness2[i2]
    return population_result, fitness_result, probability_result

def merge_sort_all(array, fitness, probability):
    if len(array) <= 1:
        return array, fitness, probability
    else:
        mid = int(len(array) / 2)
        array1, fitness[:mid], probability[:mid] = merge_sort_all(array[:mid], fitness[:mid], probability[:mid])
        array2, fitness[mid:], probability[mid:] = merge_sort_all(array[mid:], fitness[mid:], probability[mid:])
        return merge_all(array1, array2, fitness[:mid], fitness[mid:], probability[:mid], probability[mid:])

def fast_loadtxt(population_size, dimension):
    fast_population = np.loadtxt('fast_population.txt')
    fast_population = fast_population.reshape(-1, population_size, dimension)
    fast_probability = np.loadtxt('fast_probability.txt')
    fast_probability = fast_probability.reshape(-1, population_size)
    return fast_population, fast_probability
