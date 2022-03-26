from random import choices
from typing import List, Callable
import script

# 7 ACTIONS:
# 0:go right, 1:go left, 2:go up, 3:go down, 4:get object, 5:search treasure, 6:lock door

# Genome is an int, Population is a list of Genomes
Population = List[int]
FitnessFunc = Callable[[int],int]

# genetic representation of a solution
def genome_generator(length: int) -> int:
    return random.randint(0,6)

# a function to generate new solutions
def population_generator(size: int, genome_len: int) -> Population:
    return [genome_generator(genome_len) for _ in range(size)]

# fitness function: based on the amount of treasure in a certain number of moves
# TODO: consider time or end the game by finding the solution -- connect treasures to the solution
def fitness(genome: int, inventory: [script.Treasure], move_num: int) -> int:
    value = 0
    # TODO: move: number of moves
    if move > move_num:
        return 0
    for i in inventory:
        value += i.worth
    return value

# select function
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(g) for g in population],
        # draw twice from our population
        k=2
    )

# crossover function
def single_p_crossover(a: int, b: int) -> Tuple[int, int]:
    p = randint

        
