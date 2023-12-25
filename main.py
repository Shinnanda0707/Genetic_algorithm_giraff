from random import choice, choices
import time

import numpy as np


class Giraff:
    def __init__(self, height=None):
        self.height = height
        self.fitness = 0


def set_map(map_size_x, map_size_y, averages):
    maps = np.zeros((4, map_size_x, map_size_y))
    for i in averages:
        maps[i // 10 - 1] = np.random.randn(map_size_x, map_size_y) + i
    return maps


# Initial variables
n = 1225
k = 50
threshold = 3000
max_height, min_height = 50, 1
map_size_x, map_size_y = 50, 50
margin = 5
weights = [60, 35, 5]
averages = np.array(range(10, 41, 10))
it = 200

# Set initial state
individual = np.full(n, Giraff)
for i in range(individual.size):
    individual[i] = Giraff(np.random.randint(min_height, max_height))

# Evolution
max_fitness = [0, 0]
iteration = 0
heights = list()
fitnesses = list()
# while max_fitness[1] < threshold:
for i in range(it):
    st = time.time()
    # Reset map
    maps = np.zeros((4, map_size_x, map_size_y))
    for i in range(averages.size):
        maps[i] = np.random.randn(map_size_x, map_size_y) + averages[i]

    # Calculate fitness of each individual
    for i in range(individual.size):
        fitness = 0
        for map in maps:
            for x in map:
                for element in x:
                    if (individual[i].height - margin <= element) and (element <= individual[i].height + margin):
                        fitness += 1
        individual[i].fitness = fitness
    
    # Select top k individuals
    selected = np.array(sorted(individual, key=lambda indv: indv.fitness)[::-1])
    max_fitness = [selected[0].height, selected[0].fitness]
    index = 0
    for i in range(k):
        for j in range(i + 1, k):
            dec_type = choices(range(3), weights=weights)
            if dec_type == [0]:
                individual[index] = Giraff((selected[i].height + selected[j].height) // 2)
            elif dec_type == [2]:
                individual[index] = Giraff(choice([selected[i].height, selected[j].height]))
            else:
                individual[index] = Giraff(np.random.randint(min_height, max_height))
            index += 1
    iteration += 1

    heights.append(max_fitness[0])
    fitnesses.append(max_fitness[1])
    print(iteration)
    et = round(time.time() - st)
    print(f"Epoch: {i + 1}, took {et // 60}min {et % 60}sec")
print(f"{'=' * 30}[Final Model Summary]{'=' * 30}")
print(f"| Fitness:   {max_fitness[1]}")
print(f"| Height:    {max_fitness[0]}")
print(f"| Iteration: {iteration}")
print("=" * 81)

with open("data.txt", "a", encoding="UTF-8") as f:
    for i in range(iteration):
        f.write(f"{heights[i]} {fitnesses[i]}\n")
    f.close()
