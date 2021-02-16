# -*- coding: utf-8 -*-
import random
import numpy
import sys
sys.path.append('code')
import puzzleState
from strategy import Strategy


# Give names to fitness objective indices to increase code readability
CELLS_LIT = 0
BULB_CELL_VIOLATIONS = 1
BLACK_CELL_VIOLATIONS = 2
NUM_BULBS = 3


class Genotype:
    def __init__(self, encoding, p_m, fitness, fitness_mo):
        self.encoding = encoding
        self.p_m = p_m
        self.fitness = fitness # Fitness is level of non-domination
        self.fitness_mo = fitness_mo # Fitness for multiple objectives
        self.crowding_value = 0


class MOEAStrategy(Strategy):
    """
    Class implementing the MOEA strategy.
    """

    def __init__(self, experiment):
        self.fitness_function = '3_obj'
        self.fitness_sharing = False
        self.fitness_sharing_sigma = 10
        self.crowding = False
        self.initialization = 'random'
        self.ea_mu = 100
        self.ea_lambda = 50
        self.p_c = 1.00
        self.p_mchild = 0.70
        self.p_m = 0.01
        self.mutation_rate_strategy = 'fixed'
        self.parent_selection = 'fitness_proportional_selection'
        self.tournament_size_for_parent_selection = 50
        self.survival_strategy = 'plus'
        self.survival_selection = 'truncation'
        self.tournament_size_for_survival_selection = 50
        self.termination = 'number_of_evals'
        self.n_for_no_change_in_top_level = 100

        try:
            self.fitness_function = experiment.config_parser.get('moea_options',
                                                                 'fitness_function').lower()
            print('config: fitness_function =', self.fitness_function)
        except:
            print('config: fitness_function not specified; using', self.fitness_function)

        try:
            self.fitness_sharing = experiment.config_parser.getboolean('moea_options',
                                                                       'fitness_sharing')
            print('config: fitness_sharing =', self.fitness_sharing)
        except:
            print('config: fitness_sharing not specified; using', self.fitness_sharing)

        if (self.fitness_sharing):
            try:
                self.fitness_sharing_sigma = experiment.config_parser.getfloat(
                    'moea_options', 'fitness_sharing_sigma')
                print('config: fitness_sharing_sigma =',
                      self.fitness_sharing_sigma)
            except:
                print('config: fitness_sharing_sigma not specified; using',
                      self.fitness_sharing_sigma)

        try:
            self.crowding = experiment.config_parser.getboolean('moea_options',
                                                                'crowding')
            print('config: crowding =', self.crowding)
        except:
            print('config: crowding not specified; using', self.crowding)

        try:
            self.initialization = experiment.config_parser.get('moea_options',
                                                               'initialization').lower()
            print('config: initialization =', self.initialization)
        except:
            print('config: initialization not specified; using', self.initialization)

        try:
            self.ea_mu = experiment.config_parser.getint('moea_options', 'mu')
            print('config: mu =', self.ea_mu)
        except:
            print('config: mu not specified; using', self.ea_mu)

        try:
            self.ea_lambda = experiment.config_parser.getint('moea_options', 'lambda')
            print('config: lambda =', self.ea_lambda)
        except:
            print('config: lambda not specified; using', self.ea_lambda)

        try:
            self.p_c = experiment.config_parser.getfloat('moea_options', 'p_c')
            print('config: p_c =', self.p_c)
        except:
            print('config: p_c not specified; using', self.p_c)

        try:
            self.p_mchild = experiment.config_parser.getfloat('moea_options', 'p_mchild')
            print('config: p_mchild =', self.p_mchild)
        except:
            print('config: p_mchild not specified; using', self.p_mchild)

        try:
            self.p_m = experiment.config_parser.getfloat('moea_options', 'p_m')
            print('config: p_m =', self.p_m)
        except:
            print('config: p_m not specified; using', self.p_m)

        try:
            self.mutation_rate_strategy = experiment.config_parser.get('moea_options',
                                                                       'mutation_rate_strategy')
            print('config: mutation_rate_strategy =', self.mutation_rate_strategy)
        except:
            print('config: mutation_rate_strategy not specified; using', self.mutation_rate_strategy)

        try:
            self.parent_selection = experiment.config_parser.get('moea_options',
                                                                 'parent_selection').lower()
            print('config: parent_selection =', self.parent_selection)
        except:
            print('config: parent_selection not specified; using', self.parent_selection)

        if (self.parent_selection == 'k_tournament_with_replacement'):
            try:
                self.tournament_size_for_parent_selection = experiment.config_parser.getint(
                    'moea_options', 'tournament_size_for_parent_selection')
                print('config: tournament_size_for_parent_selection =',
                      self.tournament_size_for_parent_selection)
            except:
                print('config: tournament_size_for_parent_selection not specified; using',
                      self.tournament_size_for_parent_selection)

        try:
            self.survival_strategy = experiment.config_parser.get('moea_options',
                                                                  'survival_strategy').lower()
            print('config: survival_strategy =', self.survival_strategy)
        except:
            print('config: survival_strategy not specified; using', self.survival_strategy)

        try:
            self.survival_selection = experiment.config_parser.get('moea_options',
                                                                   'survival_selection').lower()
            print('config: survival_selection =', self.survival_selection)
        except:
            print('config: survival_selection not specified; using', self.survival_selection)

        if (self.survival_selection == 'k_tournament_without_replacement'):
            try:
                self.tournament_size_for_survival_selection = experiment.config_parser.getint(
                    'moea_options', 'tournament_size_for_survival_selection')
                print('config: tournament_size_for_survival_selection =',
                      self.tournament_size_for_survival_selection)
            except:
                print('config: tournament_size_for_survival_selection not specified; using',
                      self.tournament_size_for_survival_selection)

        try:
            self.termination = experiment.config_parser.get('moea_options',
                                                            'termination').lower()
            print('config: termination =', self.termination)
        except:
            print('config: termination not specified; using', self.termination)

        if (self.termination == 'no_change_in_top_level'):
            try:
                self.n_for_no_change_in_top_level = experiment.config_parser.getint('moea_options',
                                                                                    'n_for_no_change_in_top_level')
                print('config: n_for_no_change_in_top_level =',
                      self.n_for_no_change_in_top_level)
            except:
                print('config: n_for_no_change_in_top_level not specified; using',
                      self.n_for_no_change_in_top_level)

        # Dump parms to log file
        experiment.log_file.write('fitness function: ' + self.fitness_function + '\n')
        experiment.log_file.write('fitness sharing: ' + str(self.fitness_sharing) + '\n')
        if (self.fitness_sharing):
            experiment.log_file.write('fitness sharing sigma: ' + str(self.fitness_sharing_sigma) + '\n')
        experiment.log_file.write('crowding: ' + str(self.crowding) + '\n')
        experiment.log_file.write('initialization method: ' + self.initialization + '\n')
        experiment.log_file.write('mu: ' + str(self.ea_mu) + '\n')
        experiment.log_file.write('lambda: ' + str(self.ea_lambda) + '\n')
        experiment.log_file.write('p_c: ' + str(self.p_c) + '\n')
        experiment.log_file.write('p_mchild: ' + str(self.p_mchild) + '\n')
        experiment.log_file.write('p_m: ' + str(self.p_m) + '\n')
        experiment.log_file.write('mutation rate strategy: ' + self.mutation_rate_strategy + '\n')
        experiment.log_file.write('parent selection method: ' + self.parent_selection + '\n')
        if (self.parent_selection == 'k_tournament_with_replacement'):
            experiment.log_file.write('tournament size for parent selection: '
                                      + str(self.tournament_size_for_parent_selection) + '\n')
        experiment.log_file.write('survival strategy: ' + self.survival_strategy + '\n')
        experiment.log_file.write('survival selection method: ' + self.survival_selection + '\n')
        if (self.survival_selection == 'k_tournament_without_replacement'):
            experiment.log_file.write('tournament size for survival selection: '
                                      + str(self.tournament_size_for_survival_selection) + '\n')
        experiment.log_file.write('termination method: ' + self.termination + '\n')
        if (self.termination == 'no_change_in_top_level'):
            experiment.log_file.write('n evals for no_change_in_top_level: '
                                      + str(self.n_for_no_change_in_top_level) + '\n')


    def generate_random_individual(self, base_puzzle_state, experiment):
        """
        Generate a random individual. Distribute bulbs to open
        cells using uniform random.

        Returns a genotype instance.
        """
        # Pick a random number of bulbs to place
        base_puzzle_state.reset()
        num_open_cells = base_puzzle_state.count_cell_type(puzzleState.OPEN_CELL)
        num_bulbs = int(random.randint(1, num_open_cells))

        # For each bulb, loop forever until we pick an open cell.
        for bulb in range(num_bulbs):
            while True:
                i = int(random.randint(0, base_puzzle_state.rows - 1))
                j = int(random.randint(0, base_puzzle_state.cols - 1))
                if (base_puzzle_state.puzzle[i][j] == puzzleState.OPEN_CELL):
                    base_puzzle_state.puzzle[i][j] = puzzleState.BULB
                    break

        base_puzzle_state.calculate_fitness_mo()
        encoding = base_puzzle_state.get_encoding()
        return Genotype(encoding, self.p_m, 0, base_puzzle_state.fitness_mo)


    def dominates(self, individual1, individual2, four_obj):
        """
        Return whether individual1 dominates individual2.

        Individuals' crowding values used as a tiebreaker, which are always 0
        if crowding is not enabled.

        four_obj = True if using four objectives; False if using default three
        """
        anyWorse = False
        atLeastOneBetter = False

        num_obj = 3
        if (four_obj):
            num_obj = 4

        # For each objective, compare two individuals
        for i in range(num_obj):

            # If first > second, or first == second and first has better crowding
            # value, then first is better.
            if ((individual1.fitness_mo[i] > individual2.fitness_mo[i])
                or ((individual1.fitness_mo[i] == individual2.fitness_mo[i])
                    and (individual1.crowding_value > individual2.crowding_value))):
                atLeastOneBetter = True

            # If first < second, or first == second and first has worse crowding
            # value, then first is worse.
            if ((individual1.fitness_mo[i] < individual2.fitness_mo[i])
                or ((individual1.fitness_mo[i] == individual2.fitness_mo[i])
                    and (individual1.crowding_value < individual2.crowding_value))):
                anyWorse = True

        return (atLeastOneBetter and (not anyWorse))


    def calculate_levels(self, population, four_obj):
        """
        Use NSGA-II algorithm to calculate non-domination levels of population.

        Much thanks to Deb et. al, "A Fast and Elitist Multiobjective
        Genetic Algorithm: NSGA-II," https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf

        Assumes that fitness values have already been calculated for
        each member of the population.

        four_obj = True if using four objectives; False if using default three

        Returns levels
        """

        # For each member of the population, determine what others it
        # dominates, and count how many others dominate it
        dominates_lists = {}
        dominated_by_counts = {}
        level_indices = []
        level_indices.append([])
        levels = []
        levels.append([])
        for p in range(len(population)):
            dominates_lists[p] = []
            dominated_by_counts[p] = 0
            for q in range(len(population)):
               if (self.dominates(population[p], population[q], four_obj)):
                   dominates_lists[p].append(q)
               if (self.dominates(population[q], population[p], four_obj)):
                   dominated_by_counts[p] += 1
            if (dominated_by_counts[p] == 0):
                population[p].fitness = 0
                level_indices[0].append(p)
                levels[0].append(population[p])

        # Determine what population members belong to what levels (fronts).
        curr_level = 0
        while (len(levels[curr_level]) > 0):
            next_level = []
            next_level_indices = []
            for p in level_indices[curr_level]:
                for q in dominates_lists[p]:
                    dominated_by_counts[q] -= 1
                    if (dominated_by_counts[q] == 0):
                        population[q].fitness = -(curr_level + 1)
                        next_level_indices.append(q)
                        next_level.append(population[q])

            curr_level += 1
            level_indices.append(next_level_indices)
            levels.append(next_level)

        return levels


    def calculate_fitness_sharing(self, population, four_obj):
        """
        Given a population where fitness of each member = non-domination level,
        adjust those fitnesses per the fitness sharing algorithm.

        four_obj = True if using four objectives; False if using default three
        """
        # Calculate distances between each member of the population as
        # Euclidean distances of the multi-objective fitness values
        distances = numpy.zeros((len(population), len(population)))
        for i in range(len(population)):
            for j in range(len(population)):
                if (not four_obj):
                    vec1 = numpy.array(population[i].fitness_mo[0:3])
                    vec2 = numpy.array(population[j].fitness_mo[0:3])
                    distances[i][j] = numpy.linalg.norm(vec1 - vec2)
                else:
                    vec1 = numpy.array(population[i].fitness_mo)
                    vec2 = numpy.array(population[j].fitness_mo)
                    distances[i][j] = numpy.linalg.norm(vec1 - vec2)

        # Set new fitness values
        for i in range(len(population)):
            # Find out who's within sharing distance sigma and sum the share divisor
            divisor = 0
            for j in range(len(population)):
                if (distances[i][j] <= self.fitness_sharing_sigma):
                    divisor += distances[i][j]
            # Update fitness
            if (divisor > 0):
                population[i].fitness += population[i].fitness / divisor


    def crowding_selection(self, population, levels, four_obj):
        """
        Given a population where fitness of each member = non-domination level,
        adjust the crowding value of each individual per the crowding algorithm.

        Then, create the new population of size mu from the top levels.

        four_obj = True if using four objectives; False if using default three
        """
        num_obj = 3
        if (four_obj):
            num_obj = 4

        for level in levels:
            for curr_ind in level:
                curr_ind.crowding_value = 0.0
            for curr_obj in range(num_obj):
                if (len(level) == 0):
                    continue
                sorted_level = sorted(level, key = lambda i: i.fitness_mo[curr_obj])
                min_obj = sorted_level[0].fitness_mo[curr_obj]
                max_obj = sorted_level[-1].fitness_mo[curr_obj]
                sorted_level[0].crowding_value = float('inf')
                sorted_level[-1].crowding_value = float('inf')
                for i in range(1, len(sorted_level) - 1):
                    if ((max_obj - min_obj) == 0):
                        sorted_level[i].crowding_value = float('inf')
                    else:
                        prev_obj = sorted_level[i - 1].fitness_mo[curr_obj]
                        next_obj = sorted_level[i + 1].fitness_mo[curr_obj]
                        sorted_level[i].crowding_value += (next_obj - prev_obj) / (max_obj - min_obj)

        new_population = []
        for level in levels:
            if ((len(level) + len(new_population)) <= self.ea_mu):
                new_population += level
            else:
                num_needed = self.ea_mu - len(new_population)
                sorted_level = sorted(level, key = lambda i: i.crowding_value,
                                      reverse = True)
                new_population += sorted_level[0:num_needed]

        return new_population


    def is_better_front(self, front1, front2, four_obj):
        """
        Returns True if Pareto front1 is better than Pareto front2 based on
        whether the proportion of solutions in front1 which dominate at least
        one solution in front2 is larger than the proportion of solutions in
        front2 which dominate at least one solution in front1

        four_obj = True if using four objectives; False if using default three
        """

        # Null check
        if (front1 is None):
            return False
        if (front2 is None):
            return True

        one_over_two = 0
        two_over_one = 0
        for individual1 in front1:
            for individual2 in front2:
                if (self.dominates(individual1, individual2, four_obj)):
                    one_over_two += (1.0 / len(front1))
                    break
        for individual2 in front2:
            for individual1 in front1:
                if (self.dominates(individual2, individual1, four_obj)):
                    two_over_one += (1.0 / len(front2))
                    break

        return (one_over_two > two_over_one)


    def random_selection_with_replacement(self, population, num_to_select):
        """
        Return a selection made up of num_to_select randomly-
        selected individuals from the given population
        """
        selection = []
        while (len(selection) < num_to_select):
            index = random.randint(0, (len(population) - 1))
            selection.append(population[index])
        return selection


    def random_selection_without_replacement(self, population, num_to_select):
        """
        Return a selection made up of num_to_select randomly-
        selected individuals from the given population
        """
        if (len(population) < num_to_select):
            print('Stuck in random selection without replacement because',
                  len(population), 'insufficient to choose', num_to_select)

        random.shuffle(population)
        selection = population[0:num_to_select]
        return selection


    def fitness_proportional_selection(self, population, num_to_select):
        """
        Given a population, return a selection using Fitness Proportional Selection
        """
        selection = []
        probabilities = []

        # Find total fitness of population. Account for negative fitnesses
        # with an offset.
        min_fitness = population[0].fitness
        for individual in population:
            if (individual.fitness < min_fitness):
                min_fitness = individual.fitness
        offset = 0
        if (min_fitness < 0):
            offset = abs(min_fitness)
        total_fitness = 0
        for individual in population:
            total_fitness += (individual.fitness + offset)

        # If total fitness is nonzero, good to go.
        if (total_fitness != 0):
            # Calculate probability distribution
            accumulation = 0.0
            for individual in population:
                accumulation += ((individual.fitness + offset) / total_fitness)
                probabilities.append(accumulation)

            # Build new population using roulette wheel algorithm
            while (len(selection) < num_to_select):
                randval = random.random()
                curr_member = 0
                while (probabilities[curr_member] < randval):
                    curr_member += 1
                if (curr_member > (len(population) - 1)):
                    curr_member = len(population) - 1
                selection.append(population[curr_member])

        # Edge case: If total_fitness == 0, then just pick from population
        # with uniform probability, because the roulette wheel can't
        # handle having an infinite number of 0-width wedges.
        else:
            selection = self.random_selection_without_replacement(population, num_to_select)

        return selection


    def fitness_proportional_selection_without_replacement(self, population, num_to_select):
        """
        Given a population, return a selection using Fitness Proportional Selection
        without replacement
        """
        selection = []
        selected_indices = []
        probabilities = []

        # Find total fitness of population. Account for negative fitnesses
        # with an offset.
        min_fitness = population[0].fitness
        for individual in population:
            if (individual.fitness < min_fitness):
                min_fitness = individual.fitness
        offset = 0
        if (min_fitness < 0):
            offset = abs(min_fitness)
        total_fitness = 0
        for individual in population:
            total_fitness += (individual.fitness + offset)

        # If total fitness is nonzero, good to go.
        if (total_fitness != 0):
            # Calculate probability distribution
            accumulation = 0.0
            for individual in population:
                accumulation += ((individual.fitness + offset) / total_fitness)
                probabilities.append(accumulation)

            # Build new population using roulette wheel algorithm
            while (len(selection) < num_to_select):
                randval = random.random()
                curr_member = 0
                while (probabilities[curr_member] < randval):
                    curr_member += 1
                if (curr_member > (len(population) - 1)):
                    curr_member = len(population) - 1
                if (selected_indices.count(curr_member) == 0):
                    selected_indices.append(curr_member)
                    selection.append(population[curr_member])

        # Edge case: If total_fitness == 0, then just pick from population
        # with uniform probability, because the roulette wheel can't
        # handle having an infinite number of 0-width wedges.
        else:
            selection = self.random_selection_without_replacement(population, num_to_select)

        return selection


    def k_tournament_selection_with_replacement(self, population,
                                                tournament_size, num_to_select):
        """
        Given a population, return a selection using k-tournament Selection
        with replacement
        """
        selection = []

        while (len(selection) < num_to_select):
            # Pick tournament contestants
            k_contestants = []
            k_indices = []
            while (len(k_contestants) < tournament_size):
                # Pick a contestant
                selected_index = random.randint(0, len(population) - 1)
                selected_contestant = population[selected_index]
                # Add contestant if it's not already in tournament
                if (k_indices.count(selected_index) == 0):
                    k_contestants.append(selected_contestant)
                    k_indices.append(selected_index)

            # Choose the best-rated contestant.
            k_contestants = sorted(k_contestants, key=lambda item: item.fitness, reverse=True)
            selection.append(k_contestants[0])

        return selection


    def k_tournament_selection_without_replacement(self, population,
                                                   tournament_size, num_to_select):
        """
        Given a population, return a selection using k-tournament Selection
        without replacement
        """
        if (len(population) < num_to_select):
            print('Stuck in k-tournament selection without replacement because',
                  len(population), 'insufficient to choose', num_to_select)

        selection = []
        selected_indices = []

        while (len(selection) < num_to_select):
            # Pick tournament contestants
            k_contestants = []
            k_indices = []

            # If we can't fill a tournament with unselected individuals,
            # reduce tournament size to what's left
            if ((len(population) - len(selected_indices)) < tournament_size):
                tournament_size = len(population) - len(selected_indices)

            while (len(k_contestants) < tournament_size):
                # Pick a contestant
                selected_index = random.randint(0, len(population) - 1)
                selected_contestant = population[selected_index]

                # Add contestant if it's not already in tournament AND has
                # never been selected
                if ((k_indices.count(selected_index) == 0)
                    and (selected_indices.count(selected_index) == 0)):
                    k_contestants.append((selected_contestant, selected_index))
                    k_indices.append(selected_index)

            # Choose the best-rated contestant if at least one exists.
            if (len(k_contestants) > 0):
                k_contestants = sorted(k_contestants, key=lambda item: item[0].fitness, reverse=True)
                selection.append(k_contestants[0][0])
                selected_indices.append(k_contestants[0][1])

        return selection


    def stochastic_uniform_sampling(self, population, num_to_select):
        """
        Given a population, return a selection using Stochastic Uniform Sampling
        """
        selection = []
        probabilities = []

        # Find total fitness of population. Account for negative fitnesses
        # with an offset.
        min_fitness = population[0].fitness
        for individual in population:
            if (individual.fitness < min_fitness):
                min_fitness = individual.fitness
        offset = 0
        if (min_fitness < 0):
            offset = abs(min_fitness)
        total_fitness = 0
        for individual in population:
            total_fitness += (individual.fitness + offset)

        # If total fitness is nonzero, good to go.
        if (total_fitness != 0):
            # Calculate probability distribution
            accumulation = 0.0
            for individual in population:
                accumulation += ((individual.fitness + offset) / total_fitness)
                probabilities.append(accumulation)

            # Spin the roulette wheel... once... and build the new population.
            index = 0
            randval = random.random() * (1.0 / num_to_select)
            while (len(selection) < num_to_select):
                while (randval <= probabilities[index]):
                    selection.append(population[index])
                    randval += (1.0 / num_to_select)
                index += 1

        # Edge case: If total_fitness == 0, then just pick from population
        # with uniform probability, because the roulette wheel can't
        # handle having an infinite number of 0-width wedges.
        else:
            selection = self.random_selection_without_replacement(population, num_to_select)

        return selection


    def truncation_selection(self, population, num_to_select):
        """
        Truncation selection. Sort the population and select the top individuals.
        """
        sorted_population = sorted(population, key=lambda item: item.fitness, reverse=True)
        selection = []
        for curr_individual in range(self.ea_mu):
            selection.append(sorted_population[curr_individual])

        return selection


    def select_parents(self, population):
        """
        Given a population, return a mating pool using the configured method.
        """
        if (self.parent_selection == "fitness_proportional_selection"):
            return self.fitness_proportional_selection(population,
                                                       self.ea_mu)
        elif (self.parent_selection == "k_tournament_with_replacement"):
            return self.k_tournament_selection_with_replacement(population,
                                                                self.tournament_size_for_parent_selection,
                                                                self.ea_mu)
        elif (self.parent_selection == "stochastic_uniform_sampling"):
            return self.stochastic_uniform_sampling(population, self.ea_mu)
        elif (self.parent_selection == "uniform_random"):
           return self.random_selection_with_replacement(population, self.ea_mu)
        else:
            print("Unknown parent selection method:", self.parent_selection)
            sys.exit(1)


    def crossover(self, parent1, parent2):
        """
        Return a tuple of two offspring created by single-point crossover
        between the two parents at a random point (no fitnesses evaluated yet)
        """
        crossover = random.randint(0, len(parent1.encoding) - 1)
        offspring_1_encoding = []
        offspring_2_encoding = []
        for curr_gene in range(crossover):
            offspring_1_encoding.append(parent1.encoding[curr_gene])
            offspring_2_encoding.append(parent2.encoding[curr_gene])
        for curr_gene in range(crossover, len(parent2.encoding)):
            offspring_1_encoding.append(parent2.encoding[curr_gene])
            offspring_2_encoding.append(parent1.encoding[curr_gene])
        offspring_1 = Genotype(offspring_1_encoding, parent1.p_m, 0, None)
        offspring_2 = Genotype(offspring_2_encoding, parent2.p_m, 0, None)
        return (offspring_1, offspring_2)


    def recombine_population(self, population):
        """
        Given a population, execute recombination strategy and return
        offspring
        """
        offspring = []

        # Create mating pool
        mating_pool = self.select_parents(population)

        # Shuffle mating pool
        mating_pool = self.random_selection_without_replacement(mating_pool,
                                                                len(mating_pool))

        index = 0
        # Recombine parents of different encodings pairwise
        while (len(offspring) < self.ea_lambda):
            parent1 = mating_pool[index]
            index += 1
            parent2 = mating_pool[index]
            index += 1

            # Each pair has a p_c chance of mating. p_c defaults to 1.0.
            if ((random.random() < self.p_c)
                and (parent1.encoding != parent2.encoding)):
                new_offspring = self.crossover(parent1, parent2)
                offspring.append(new_offspring[0])
                if (len(offspring) < self.ea_lambda):
                    offspring.append(new_offspring[1])
            # Sometimes we end up with a population of all
            # the same encoding. I can either carry them
            # on to the next generation, or just sit in this loop forever.
            else:
                offspring.append(parent1)
                if (len(offspring) < self.ea_lambda):
                    offspring.append(parent2)

        return offspring


    def mutate_individual_offspring(self, individual):
        """
        Flip each bit with probability in mutation rate.
        """
        # For Self-Adaptive Mutation Rate, mutate the mutation rate.
        # Use uncorreleted mutation with one step size.
        if (self.mutation_rate_strategy == 'self_adaptive'):
            # Update mutation rate
            tau = 1.0 / (len(individual.encoding) ** 0.5)
            individual.p_m *= numpy.exp(tau * numpy.random.normal(0, 1))
            # Threshold
            if (individual.p_m < 0):
                individual.p_m = 0
            if (individual.p_m > 1.0):
                individual.p_m = 1.0

        for curr_bit in range(len(individual.encoding)):
            if (random.random() < individual.p_m):
                individual.encoding[curr_bit] = int(not individual.encoding[curr_bit])


    def mutate_offspring(self, offspring):
        """
        Process every offspring for possible mutation.
        """
        for curr_child in offspring:
            if (random.random() < self.p_mchild):
                self.mutate_individual_offspring(curr_child)
        return offspring


    def select_survivors(self, population):
        """
        Given a population, return a selection of survivors using the
        configured method.
        """
        if (self.survival_selection == 'truncation'):
            return self.truncation_selection(population, self.ea_mu)
        elif (self.survival_selection == 'k_tournament_without_replacement'):
            return self.k_tournament_selection_without_replacement(population,
                                                                   self.tournament_size_for_survival_selection,
                                                                   self.ea_mu)
        elif (self.survival_selection == 'uniform_random_without_replacement'):
            return self.random_selection_without_replacement(population, self.ea_mu)
        elif (self.survival_selection == 'fitness_proportional_selection_without_replacement'):
            return self.fitness_proportional_selection_without_replacement(population, self.ea_mu)
        else:
            print('Unknown survival selection method:', self.survival_selection)
            sys.exit(1)


    def format_fitness_mo(self, avg_fitness_mo, best_fitness_mo, four_obj):
        """
        Given two MO fitness arrays, return a string of the format required
        for the log.

        four_obj = True if using four objectives; False if using default three
        """
        return_me = str(avg_fitness_mo[CELLS_LIT]) + '\t' + str(best_fitness_mo[CELLS_LIT]) + '\t'
        return_me += str(avg_fitness_mo[BULB_CELL_VIOLATIONS]) + '\t' + str(best_fitness_mo[BULB_CELL_VIOLATIONS]) + '\t'
        return_me += str(avg_fitness_mo[BLACK_CELL_VIOLATIONS]) + '\t' + str(best_fitness_mo[BLACK_CELL_VIOLATIONS])
        if (four_obj):
            return_me +=  '\t' + str(avg_fitness_mo[NUM_BULBS]) + '\t' + str(best_fitness_mo[NUM_BULBS])
        return return_me


    def execute_one_run(self, working_puzzle_state, experiment):
        """
        Given a puzzle state and experimental parameters, execute a single run.
        """
        # Population for this run will be a list of Genotype instances
        population = []

        # Bookkeeping variables
        eval_count = 0
        evals_with_no_change = 0

        # Initialization -- reset puzzle
        working_puzzle_state.reset()

        # Set unique bulbs in puzzle state if "validity forced"
        if (self.initialization == 'validity_forced'):
            working_puzzle_state.place_unique_bulbs()

        # Set initial best state
        working_puzzle_state.calculate_fitness_mo()
        best_fitness_mo = working_puzzle_state.fitness_mo

        # Create the initial population (size mu)
        for individual_num in range(self.ea_mu):
            individual = self.generate_random_individual(working_puzzle_state,
                                                         experiment)
            eval_count += 1 # generate_random_individual does an eval
            population.append(individual)

        # Calculate levels
        levels = self.calculate_levels(population,
                                       self.fitness_function == '4_obj')

        # Create initial "last" and "best" Pareto fronts
        last_pareto_front = levels[0]
        best_pareto_front = last_pareto_front

        # Run generation after generation until we hit termination condition
        # and break out of the loop
        while (True):

            # Calculate log stats
            population_total_fitness_mo = [0, 0, 0, 0]
            for individual in population:
                population_total_fitness_mo = [sum(i) for i in zip(population_total_fitness_mo,
                                                                   individual.fitness_mo)]
                best_fitness_mo = numpy.maximum(best_fitness_mo, individual.fitness_mo)

            # Update log file
            experiment.log_file.write(str(eval_count) + '\t'
                                      + self.format_fitness_mo([(x / len(population)) for x in population_total_fitness_mo],
                                                               best_fitness_mo,
                                                               self.fitness_function == '4_obj')
                                      + str() + '\n')

            # Check for termination
            if (self.termination == 'number_of_evals'):
                if (eval_count >= experiment.num_fitness_evals_per_run):
                    break
            elif (self.termination == 'no_change_in_top_level'):
                if (evals_with_no_change >= self.n_for_no_change_in_top_level):
                    print('CONVERGED at', eval_count, 'evals')
                    break
            else:
                print('Unknown termination method:', self.termination)
                sys.exit(1)

            # Not terminating? Let's proceed!

            # Recombine
            offspring = self.recombine_population(population)

            # Mutate
            offspring = self.mutate_offspring(offspring)

            # If this is a generational EA, wipe out the old population
            if (self.survival_strategy == 'comma'):
                population = []   # I have faith in the Python garbage collector

            # Add to offspring to population and evaluate fitness
            for curr_child in offspring:
                working_puzzle_state.decode_state_from(curr_child.encoding)
                working_puzzle_state.calculate_fitness_mo()
                eval_count += 1
                curr_child.fitness_mo = working_puzzle_state.fitness_mo
                population.append(curr_child)
                # Provide status message every nth evaluation.
                if ((eval_count % 1000) == 0):
                    print('\r', eval_count, 'evals', end =" ")

            # Find the Pareto Front
            levels = self.calculate_levels(population,
                                           self.fitness_function == '4_obj')

            # Check for changes in the members of the Pareto front
            # since the last generation.
            if (set(levels[0]) == set(last_pareto_front)):
                evals_with_no_change += 1
            else:
                evals_with_no_change = 0
                last_pareto_front = levels[0]

            # Check for best Pareto front so far
            if (self.is_better_front(last_pareto_front, best_pareto_front,
                                     self.fitness_function == '4_obj')):
                best_pareto_front = last_pareto_front

            # Calculate fitness sharing values if enabled
            if (self.fitness_sharing):
                self.calculate_fitness_sharing(population,
                                               self.fitness_function == '4_obj')

            # Use crowding to select survivors, if enabled
            if (self.crowding):
                population = self.crowding_selection(population, levels,
                                                     self.fitness_function == '4_obj')
            else:
                # Use more traditional survival selection method
                population = self.select_survivors(population)



        # End of the run. Print some stats.
        print()
        print('Size of Pareto front this run:', len(best_pareto_front))
        print('average mo:',
              [(x / len(population)) for x in population_total_fitness_mo],
              end = ' ')
        print('/ best mo:', best_fitness_mo)

        return best_pareto_front
