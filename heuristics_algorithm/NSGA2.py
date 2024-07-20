import random
import numpy as np
import math
import copy

from typing import Optional

import matplotlib.pyplot as plt

# 定义问题参数
POP_SIZE = 50  # 种群大小
GENERATIONS = 5000  # 迭代次数
CROSSOVER_RATE = 0.8  # 交叉率
MUTATION_RATE = 0.2  # 突变率
NUM_VARIABLES = 30 # 基因数量


# 个体
class Individual:
    def __init__(self, gene: list = []) -> None:
        self.gene = gene
        self.fitness = [] # could be multi-objective results
        self.rank: Optional[int] = None
        self.crowding_distance: Optional[float] = None


# 定义问题：示例问题是一个简单的多目标优化问题，目标函数有两个：f1(x) = x**2, f2(x) = (x-2)**2
def objective_function(x):
    n = len(x)
    f1 = x[0]
    g = 1 + (9 / (n - 1)) * sum(x[1:])
    h = 1 - math.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


# 交叉操作：单点交叉
def crossover(parent1: Individual, parent2: Individual):
    crossover_point = random.randint(1, len(parent1.gene) - 1)
    child1 = Individual(gene=parent1.gene[:crossover_point] + parent2.gene[crossover_point:])
    child2 = Individual(gene=parent2.gene[:crossover_point] + parent1.gene[crossover_point:])
    return child1, child2


# 突变操作：随机变异
def mutate(chromosome):
    mutated_chromosome = copy.deepcopy(chromosome)
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            mutated_chromosome[i] = random.random() 
    return mutated_chromosome


# 选择操作：锦标赛选择
def tournament_selection(population: list[Individual], num_of_parents:int):
    """
    Parameters
    ----------
    population: list[Individual]
        The population that we chose parents from.
    num_of_parents: int
        num of returned parents.

    Return
    ------
    parents:list[Individual]
    """
    selected_parents = []
    while len(selected_parents) < num_of_parents:
        tournament = random.sample(population, 2)
        # Perform non-dominated sorting
        highest_rank = min(tournament, key=lambda x: x.rank).rank
        first_stage_winners: list[Individual] = [p for p in tournament if p.rank == highest_rank]
        # Choose the front with the highest rank
        # If there are multiple individuals in the highest front, use crowding distance
        if len(first_stage_winners) > 1:
            # Choose the individual with the highest crowding distance
            final_winner = max(first_stage_winners, key=lambda x:x.crowding_distance)
            selected_parents.append(final_winner)
        elif len(first_stage_winners) == 1:
            selected_parents.extend(first_stage_winners)
        else:
            raise(ValueError("Reached the unreachable area."))
    return selected_parents


# 快速非支配排序
# 在这里划分等级及支配
def fast_non_dominated_sort(population: list[Individual]):
    """
    Return
    ------
    a list of fronts, where each front is itself a list containing the indices of individuals belonging to that front. 
    Each front contains individuals that are non-dominated by any other individual in the population.
    """
    fronts = []  # List to store fronts
    S = [[] for _ in range(len(population))]  # List to store dominated individuals for each individual
    n = [0 for _ in range(len(population))]  # List to store domination count for each individual
    fronts.append([])  # Initialize the first front
    
    # Iterate through each individual in the population
    for i in range(len(population)):
        # Initialize domination count and set of dominated individuals
        n[i] = 0
        S[i] = []
        
        # Compare individual with every other individual
        for j in range(len(population)):
            if i == j: # itself
                continue
            
            # Check dominance relation
            if dominates(population[i], population[j]):
                S[i].append(j)  # Individual i dominates individual j
            elif dominates(population[j], population[i]):
                n[i] += 1  # Individual j dominates individual i
        
        # If no individual dominates individual i, assign it to the first front
        if n[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # Initialize front index
    i = 0
    
    # Loop to create subsequent fronts
    while len(fronts[i]) > 0:
        Q = []  # Queue to store individuals for the next front
        
        # Iterate through individuals in the current front
        for p in fronts[i]:
            # Iterate through individuals dominated by p
            for q in S[p]:
                n[q] -= 1  # Decrease domination count
                if n[q] == 0:
                    population[q].rank = i + 1
                    Q.append(q)  # Add individual to the queue if it becomes non-dominated
        
        # Move to the next front
        i += 1
        fronts.append(Q)
    
    # Remove empty fronts
    fronts.pop() # last list must be empty
    
    return fronts


def dominates(x, y):
    # Function to check if individual x dominates individual y
    return all(xi <= yi for xi, yi in zip(x.fitness, y.fitness)) and any(xi < yi for xi, yi in zip(x.fitness, y.fitness))


# 拥挤度计算
def crowding_distance_assignment(population: list[Individual]):
    """Calculate crowding distance value of the same front population.

    Parameters
    ----------
    population: list[Individual]
        Individual objects of the same front population.
    """

    num_individuals = len(population)
    num_objectives = len(population[0].fitness)
    indics = [i for i in range(0, num_individuals)]
    
    # Calculate crowding distance for each objective
    for obj_index in range(num_objectives):
        # Sort individuals based on the current objective within the front
        sorted_front = sorted(indics, key=lambda x: population[x].fitness[obj_index])
        
        # Set infinite crowding distance for boundary individuals
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')
        distance_between_max_min = (population[sorted_front[-1]].fitness[obj_index] - population[sorted_front[0]].fitness[obj_index])
        # Calculate crowding distance for internal individuals
        for i in range(1, num_individuals - 1):
            distance_between_front_back = (population[sorted_front[i+1]].fitness[obj_index] - population[sorted_front[i-1]].fitness[obj_index])
            crowding_distance_value = distance_between_front_back / distance_between_max_min
            population[sorted_front[i]].crowding_distance = crowding_distance_value


# 生成个体
def generate_individuals(num=0, num_variables:int=10) -> list[Individual]:
    individuals = []
    for i in range(num):
        individual = Individual(gene=[random.random() for _ in range(num_variables)])
        individual.fitness = calculate_fitness(individual=individual)
        individuals.append(individual)
    return individuals


# 计算适应度值（目标函数值）
def calculate_fitness(individual: Individual):
    return objective_function(individual.gene)
    

def sort_rank_and_calculate_crowding(population: list[Individual]):
    fronts = fast_non_dominated_sort(population=population)
    for front in fronts:
        front_individuals = [population[i] for i in front]
        crowding_distance_assignment(population=front_individuals)
    for individual in population:
        if individual.crowding_distance == None or individual.rank == None:
            raise(ValueError("After sorting rank and calculate crowding distance, some individuals still don't have the value."))
    return fronts


# NSGA-II 主函数
def nsga2():
    population = generate_individuals(num=POP_SIZE, num_variables=NUM_VARIABLES)
    sort_rank_and_calculate_crowding(population=population)
    for iteration in range(GENERATIONS):
        offspring_population = []
        for _ in range(int(POP_SIZE / 2)):
            parent1, parent2 = tournament_selection(population=population, num_of_parents=2) # 选取一对父母
            child1, child2 = crossover(parent1, parent2)
            child1.gene = mutate(child1.gene)
            child2.gene = mutate(child2.gene)
            child1.fitness = calculate_fitness(individual=child1)
            child2.fitness = calculate_fitness(individual=child2)
            offspring_population.extend([child1, child2])
        
        combined_population = population + offspring_population
        fronts = sort_rank_and_calculate_crowding(combined_population)
        new_population = []
        front_index = 0
        # Firstly, choose new individuals based on rank.
        print(f"Num of F1 front: {len(fronts[front_index])}. Iteration: {iteration}")
        while len(new_population) + len(fronts[front_index]) <= POP_SIZE:
            individuals_of_the_same_rank = [combined_population[i] for i in fronts[front_index]]
            new_population.extend(individuals_of_the_same_rank)
            front_index += 1
        sorted_last_front = sorted(fronts[front_index], key=lambda x: combined_population[x].crowding_distance, reverse=True)
        new_population.extend([combined_population[i] for i in sorted_last_front[:POP_SIZE - len(new_population)]])
        population = new_population

    return population


def draw(x: list, y: list):
    # Create scatter plot
    if len(x) != len(y):
        raise(ValueError(f"data error. len of x is {len(x)}, and of y is {len(y)}"))
    for xi, yi in zip(x, y):
        plt.scatter(xi, yi)

    # Add labels and title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot')

    # Show plot
    plt.show()


if __name__ == "__main__":
    # 运行 NSGA-II 算法
    final_population: list[Individual] = nsga2()
    if len(final_population) > 0:
        num_of_object_funtcion = len(final_population[0].fitness)
    object_function_results = []
    object_function_results.extend([[] for _ in range(0, num_of_object_funtcion)])
    for individual in final_population:
        if len(individual.fitness) != num_of_object_funtcion:
            raise(ValueError(f"An individual's objective function went wrong. It has {len(individual.fitness)} objective funtcions, but the first individual has {num_of_object_funtcion}."))
        for i in range(0, len(individual.fitness)):
            object_function_results[i].append(individual.fitness[i])
    
    draw(x=object_function_results[0], y=object_function_results[1])

