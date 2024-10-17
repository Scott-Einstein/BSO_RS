from bso import bso
from fitness_function import rastrigin, sphere, rosenbrock, ackley, griewank, schwefel, easom, penalty1, levy_n13

def main():
    population_size = 2000 
    dimension = 10
    cluster_number = 10
    left_range = -100
    right_range = 100
    max_iteration = 2 
    for seed in range(2): # 50次BSO数据 2000*2*50
        bso(rastrigin, population_size, dimension, cluster_number, left_range, right_range, max_iteration, seed)

if __name__ == "__main__":
    main()