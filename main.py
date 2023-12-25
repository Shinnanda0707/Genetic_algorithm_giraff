from random import choice, choices
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

heights = list()
fitnesses = list()

class Giraff:
    def __init__(self, height=None):
        self.height = height
        self.fitness = 0


# Initial variables
n = 1225
k = 50
max_height, min_height = 50, 1
map_size = 20000
margin = 5
weights = [60, 35, 5]
average = 45
it = 200

# Set initial state
individual = np.full(n, Giraff)
height_dt = np.random.rand(n) + 12.8 # Activate when Experiment 2
for i in range(n):
    individual[i] = Giraff(height_dt[i])

# Evolution
max_fitness = [0, 0]
iteration = 0
heights = list()
fitnesses = list()

tb = SummaryWriter()
for _ in range(it):
    st = time.time()
    total_height, total_fitness = 0, 0

    # Reset map
    map = np.random.randn(map_size) + average

    # Calculate fitness of each individual
    for i in range(n):
        fitness = 0
        for element in map:
            if (individual[i].height - margin <= element) and (element <= individual[i].height + margin):
                fitness += 1
        individual[i].fitness = fitness
        total_height += individual[i].height
        total_fitness += fitness
    total_height /= n
    total_fitness /= n
    
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

    tb.add_scalar("Best Fitness Height", max_fitness[0], iteration)
    tb.add_scalar("Best Fitness", max_fitness[1], iteration)
    tb.add_scalar("Average Height", total_height, iteration)
    tb.add_scalar("Average Fitness", total_fitness, iteration)

    et = round(time.time() - st)
    print(f"Epoch: {iteration}, took {et // 60}min {et % 60}sec")

tb.close()
print(f"{'=' * 30}[Final Model Summary]{'=' * 30}")
print(f"| Fitness:   {max_fitness[1]}")
print(f"| Height:    {max_fitness[0]}")
print(f"| Iteration: {iteration}")
print("=" * 81)
print(total_height)
