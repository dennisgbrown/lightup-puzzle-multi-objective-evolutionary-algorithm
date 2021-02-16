# -*- coding: utf-8 -*-
import random
import copy
import numpy
import sys
sys.path.append('code')
import puzzleState
from strategy import Strategy


class Genotype:
    def __init__(self, encoding, p_m,
                 strict_fitness, fitness, is_valid):
        self.encoding = encoding
        self.p_m = p_m
        self.strict_fitness = strict_fitness
        self.fitness = fitness
        self.is_valid = is_valid


class EAStrategy(Strategy):
    """
    Class implementing the very configurable EA strategy.
    """

    def __init__(self, experiment):
        self.fitness_function = 'original'
        self.with_penalty = False
        self.c_p = 1.0
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
        self.n_for_convergence = 10
        self.log_values = 'working'
        self.generated_individuals = 0

        try:
            self.fitness_function = experiment.config_parser.get('ea_options',
                                                                 'fitness_function').lower()
            print('config: fitness_function =', self.fitness_function)
        except:
            print('config: fitness_function not specified; using', self.fitness_function)
        if (self.fitness_function == 'constraint_satisfaction'):
            self.with_penalty = True

        try:
            self.c_p = experiment.config_parser.getfloat('ea_options', 'c_p')
            print('config: c_p =', self.c_p)
        except:
            print('config: c_p not specified; using', self.c_p)

        try:
            self.initialization = experiment.config_parser.get('ea_options',
                                                               'initialization').lower()
            print('config: initialization =', self.initialization)
        except:
            print('config: initialization not specified; using', self.initialization)

        try:
            self.ea_mu = experiment.config_parser.getint('ea_options', 'mu')
            print('config: mu =', self.ea_mu)
        except:
            print('config: mu not specified; using', self.ea_mu)

        try:
            self.ea_lambda = experiment.config_parser.getint('ea_options', 'lambda')
            print('config: lambda =', self.ea_lambda)
        except:
            print('config: lambda not specified; using', self.ea_lambda)

        try:
            self.p_c = experiment.config_parser.getfloat('ea_options', 'p_c')
            print('config: p_c =', self.p_c)
        except:
            print('config: p_c not specified; using', self.p_c)

        try:
            self.p_mchild = experiment.config_parser.getfloat('ea_options', 'p_mchild')
            print('config: p_mchild =', self.p_mchild)
        except:
            print('config: p_mchild not specified; using', self.p_mchild)

        try:
            self.p_m = experiment.config_parser.getfloat('ea_options', 'p_m')
            print('config: p_m =', self.p_m)
        except:
            print('config: p_m not specified; using', self.p_m)

        try:
            self.mutation_rate_strategy = experiment.config_parser.get('ea_options',
                                                                       'mutation_rate_strategy')
            print('config: mutation_rate_strategy =', self.mutation_rate_strategy)
        except:
            print('config: mutation_rate_strategy not specified; using', self.mutation_rate_strategy)

        try:
            self.parent_selection = experiment.config_parser.get('ea_options',
                                                                 'parent_selection').lower()
            print('config: parent_selection =', self.parent_selection)
        except:
            print('config: parent_selection not specified; using', self.parent_selection)

        if (self.parent_selection == 'k_tournament_with_replacement'):
            try:
                self.tournament_size_for_parent_selection = experiment.config_parser.getint(
                    'ea_options', 'tournament_size_for_parent_selection')
                print('config: tournament_size_for_parent_selection =',
                      self.tournament_size_for_parent_selection)
            except:
                print('config: tournament_size_for_parent_selection not specified; using',
                      self.tournament_size_for_parent_selection)

        try:
            self.survival_strategy = experiment.config_parser.get('ea_options',
                                                                  'survival_strategy').lower()
            print('config: survival_strategy =', self.survival_strategy)
        except:
            print('config: survival_strategy not specified; using', self.survival_strategy)

        try:
            self.survival_selection = experiment.config_parser.get('ea_options',
                                                                   'survival_selection').lower()
            print('config: survival_selection =', self.survival_selection)
        except:
            print('config: survival_selection not specified; using', self.survival_selection)

        if (self.survival_selection == 'k_tournament_without_replacement'):
            try:
                self.tournament_size_for_survival_selection = experiment.config_parser.getint(
                    'ea_options', 'tournament_size_for_survival_selection')
                print('config: tournament_size_for_survival_selection =',
                      self.tournament_size_for_survival_selection)
            except:
                print('config: tournament_size_for_survival_selection not specified; using',
                      self.tournament_size_for_survival_selection)

        try:
            self.termination = experiment.config_parser.get('ea_options',
                                                            'termination').lower()
            print('config: termination =', self.termination)
        except:
            print('config: termination not specified; using', self.termination)

        if (self.termination == 'convergence'):
            try:
                self.n_for_convergence = experiment.config_parser.getint('ea_options',
                                                                         'n_for_convergence')
                print('config: n_for_convergence =', self.n_for_convergence)
            except:
                print('config: n_for_convergence not specified; using', self.n_for_convergence)

        try:
            self.log_values = experiment.config_parser.get('ea_options',
                                                           'log_values').lower()
            print('config: log_values =', self.log_values)
        except:
            print('config: log_values not specified; using', self.log_values)

        # Dump parms to log file
        experiment.log_file.write('fitness function: ' + self.fitness_function + '\n')
        if (self.fitness_function == 'constraint_satisfaction'):
            experiment.log_file.write('c_p: ' + str(self.c_p) + '\n')
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
        if (self.termination == 'convergence'):
            experiment.log_file.write('n evals for convergence: '
                                      + str(self.n_for_convergence) + '\n')
        experiment.log_file.write('log values: ' + self.log_values + '\n')


    def generate_random_individual(self, base_puzzle_state, working_puzzle_state, experiment):
        num_open_cells = base_puzzle_state.count_cell_type(puzzleState.OPEN_CELL)
        working_puzzle_state.reset_to(base_puzzle_state)
        """
        Generate a random individual. Returns a genotype instance.
        """
        # Distribute bulbs to open cells using uniform random
        # Pick a random number of bulbs to place
        num_bulbs = int(random.randint(1, num_open_cells))
        # For each bulb, loop forever until we pick an open cell.
        for bulb in range(num_bulbs):
            while True:
                i = int(random.randint(0, working_puzzle_state.rows - 1))
                j = int(random.randint(0, working_puzzle_state.cols - 1))
                if (working_puzzle_state.puzzle[i][j] == puzzleState.OPEN_CELL):
                    working_puzzle_state.puzzle[i][j] = puzzleState.BULB
                    break

        working_puzzle_state.calculate_fitness(experiment.enforce_black_cell_constraint,
                                               self.with_penalty,
                                               self.c_p)
        encoding = working_puzzle_state.get_encoding()
        return Genotype(encoding, self.p_m,
                        working_puzzle_state.strict_fitness,
                        working_puzzle_state.fitness,
                        working_puzzle_state.is_valid)


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
        selection = population[0:(num_to_select - 1)]
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
        offspring_1 = Genotype(offspring_1_encoding, parent1.p_m, 0, 0, False)
        offspring_2 = Genotype(offspring_2_encoding, parent2.p_m, 0, 0, False)
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


    def execute_one_run(self, working_puzzle_state, experiment):
        """
        Given a puzzle state and experimental parameters, execute a single run.
        """
        # Population for this run will be a list of Genotype instances
        population = []

        # bookkeeping variables
        eval_count = 0
        evals_with_no_change = 0
        best_puzzle_state = None
        best_fitness = float('-inf')

        # Initialization -- reset puzzle
        working_puzzle_state.reset()
        base_puzzle_state = copy.deepcopy(working_puzzle_state)

        # Set unique bulbs in puzzle state if "validity forced"
        if (self.initialization == 'validity_forced'):
            base_puzzle_state.place_unique_bulbs()
            base_puzzle_state.calculate_fitness(experiment.enforce_black_cell_constraint,
                                                self.with_penalty,
                                                self.c_p)
            # print('State after validity_forced with fitness:', base_puzzle_state.fitness)
            # base_puzzle_state.print_puzzle_ANSI_text()
            best_puzzle_state = copy.deepcopy(base_puzzle_state)
            best_fitness = best_puzzle_state.fitness

        # Create the initial population (size mu)
        population_total_strict_fitness = 0
        population_total_fitness = 0
        for individual_num in range(self.ea_mu):
            individual = self.generate_random_individual(base_puzzle_state,
                                                         working_puzzle_state,
                                                         experiment)
            if (individual.fitness > best_fitness):
                best_fitness = individual.fitness
                working_puzzle_state.decode_state_from(individual.encoding)
                best_puzzle_state = copy.deepcopy(working_puzzle_state)
                best_puzzle_state.strict_fitness = individual.strict_fitness
                best_puzzle_state.fitness = individual.fitness
                best_puzzle_state.is_valid = individual.is_valid

            population.append(individual)
            population_total_strict_fitness += individual.strict_fitness
            population_total_fitness += individual.fitness
            eval_count += 1

        # Update log file
        if (self.log_values == 'strict'):
            experiment.log_file.write(str(eval_count) + '\t'
                                      + str(population_total_strict_fitness / len(population)) + '\t'
                                      + str(best_puzzle_state.strict_fitness) + '\n')
        else:
            experiment.log_file.write(str(eval_count) + '\t'
                                      + str(population_total_fitness / len(population)) + '\t'
                                      + str(best_puzzle_state.fitness) + '\n')

        # Run generation after generation until we hit termination condition
        # and break out of the loop
        while (True):

            # Recombine
            offspring = self.recombine_population(population)

            # Mutate
            offspring = self.mutate_offspring(offspring)

            # If this is a generational EA, wipe out the old population
            if (self.survival_strategy == 'comma'):
                population = []   # I have faith in the Python garbage collector

            # Calculate fitness of offspring and add to population
            for curr_child in range(len(offspring)):
                working_puzzle_state.decode_state_from(offspring[curr_child].encoding)
                working_puzzle_state.calculate_fitness(experiment.enforce_black_cell_constraint,
                                                       self.with_penalty,
                                                       self.c_p)
                offspring[curr_child].strict_fitness = working_puzzle_state.strict_fitness
                offspring[curr_child].fitness = working_puzzle_state.fitness
                offspring[curr_child].is_valid = working_puzzle_state.is_valid
                population.append(offspring[curr_child])
                # Update termination variables
                eval_count += 1
                if (working_puzzle_state.fitness <= best_fitness):
                    evals_with_no_change += 1
                else:
                    evals_with_no_change = 0
                # Provide status message every nth evaluation.
                if ((eval_count % 1000) == 0):
                    print('\r', eval_count, 'evals', end =" ")

            # Choose survivors from (over)population
            population = self.select_survivors(population)

            # Check for best so far and save a copy
            population_total_strict_fitness = 0
            population_total_fitness = 0
            for individual in population:
                population_total_strict_fitness += individual.strict_fitness
                population_total_fitness += individual.fitness
                if (individual.fitness > best_fitness):
                    best_fitness = individual.fitness
                    working_puzzle_state.decode_state_from(individual.encoding)
                    best_puzzle_state = copy.deepcopy(working_puzzle_state)
                    best_puzzle_state.strict_fitness = individual.strict_fitness
                    best_puzzle_state.fitness = individual.fitness
                    best_puzzle_state.is_valid = individual.is_valid

            # Update log file
            if (self.log_values == 'strict'):
                experiment.log_file.write(str(eval_count) + '\t'
                                          + str(population_total_strict_fitness / len(population)) + '\t'
                                          + str(best_puzzle_state.strict_fitness) + '\n')
            else:
                experiment.log_file.write(str(eval_count) + '\t'
                                          + str(population_total_fitness / len(population)) + '\t'
                                          + str(best_puzzle_state.fitness) + '\n')

            # Check for termination
            if (self.termination == 'number_of_evals'):
                if (eval_count >= experiment.num_fitness_evals_per_run):
                    break
            elif (self.termination == 'convergence'):
                if (evals_with_no_change >= self.n_for_convergence):
                    print('CONVERGED at', eval_count, 'evals')
                    break
            else:
                print('Unknown termination method:', self.termination)
                sys.exit(1)

        print()
        print('average: ' + str(population_total_fitness / len(population)), end = ' ')
        print('best: ' + str(best_fitness), 'valid?', best_puzzle_state.is_valid, end = ' ')
        print('dist from optimal: ' + str(best_puzzle_state.max_possible_fitness - best_puzzle_state.strict_fitness))
        # best_puzzle_state.print_puzzle_ANSI_text()

        return best_puzzle_state
