from heur import Heuristic, StopCriterion
import numpy as np


class GeneticOptimization(Heuristic):

    # def __init__(self, of, maxeval, N, mutation, crossover):
    def __init__(self, of, maxeval, N):
        Heuristic.__init__(self, of, maxeval)

        # assert M > N, 'M should be larger than N'
        self.N = N  # population size
        self.P = int(N/2)  # parents' size
        # self.mutation = mutation
        # self.crossover = crossover

    # randomly sets parameters values for a
    def create_population(self, pop_size):
        population = np.zeros(shape=pop_size)
        for individ in population:
            individ[0] = np.random.uniform(low=0.125, high=4.125)
            individ[1] = np.random.uniform(low=0.125, high=4.125)
            individ[2] = np.random.randint(4, 32)
            individ[3] = np.random.randint(32, 128)
            individ[4] = np.random.randint(2, 32)
        return population

    def fitness_calc(self, new_population):
        fitness = []
        for ind in new_population:
            fitness.append(self.of.evaluate(p=ind[0], q=ind[1], num_walks=int(ind[2]), len_walks=int(ind[3]),
                                        window=int(ind[4])))
        return fitness

    @staticmethod
    def select_parents(new_population, fitness, parents_num):
        parents = np.empty((parents_num, new_population.shape[1]))
        for num in range(parents_num):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[num, :] = new_population[max_fitness_idx, :]
            fitness[max_fitness_idx] = 0
        return parents

    # 1-point crossover (1/2)
    def my_crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Here, it is at the center.
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def my_mutation(self, offspring_crossover, num_mutations=1):
        mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = np.random.randint(0, 4)
            for mutation_num in range(num_mutations):
                # The random value to be added to the gene.
                if gene_idx == 0 or gene_idx == 1:
                    random_value = np.random.uniform(-0.1, 1.0, 1)
                    offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                    # n2v parameters can't be negative
                    if(offspring_crossover[idx, gene_idx]<=0): offspring_crossover[idx, gene_idx] = 0.1
                elif gene_idx == 2 or gene_idx == 3 or gene_idx == 4:
                    random_value = np.random.randint(-10, 10, 1)
                    offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                    # n2v parameters can't be negative
                    if(offspring_crossover[idx, gene_idx]<=0): offspring_crossover[idx, gene_idx] = 1
                gene_idx = gene_idx + mutations_counter
        return offspring_crossover


    def search(self):
        try:
            pop_size = (self.N, 5)
            # generate the population
            new_population = self.create_population(pop_size)

            # Evolution iteration
            while True:
                for generation in range(self.N):

                    # Measing the fitness of each individ in the population.
                    fitness = self.fitness_calc(new_population)

                    # Selecting the best parents in the population for mating.
                    parents = self.select_parents(new_population, fitness,
                                             self.P)

                    # Generating next generation using crossover.
                    offspring_crossover = self.my_crossover(parents,
                                                    offspring_size=(pop_size[0] - parents.shape[0], 5))

                    # Adding some variations to the offsrping using mutation.
                    offspring_mutation = self.my_mutation(offspring_crossover)

                    # Creating the new population based on the parents and offspring.
                    new_population[0:parents.shape[0], :] = parents
                    new_population[parents.shape[0]:, :] = offspring_mutation

                    # calculate heur.evaluate(x) for each individual in the new population
                    for ind in new_population:
                        self.evaluate(ind)

        except StopCriterion:
            return self.report_end()
        except:
            raise
