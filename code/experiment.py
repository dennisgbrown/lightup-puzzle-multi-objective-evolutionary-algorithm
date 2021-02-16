# -*- coding: utf-8 -*-
import configparser
import random
import time
import copy
import traceback
import sys

sys.path.append('code')
from randomStrategy import RandomStrategy
from eaStrategy import EAStrategy
from moeaStrategy import MOEAStrategy

class Experiment:
    """
    Provide capabilities to run an experiment within given
    configuration parameters.
    """

    def __init__(self):
        self.random_seed = None
        self.search_strategy_name = 'random'
        self.search_strategy = None
        self.num_runs_per_experiment = 1
        self.num_fitness_evals_per_run = 10000
        self.enforce_black_cell_constraint = True
        self.log_file_path = 'logs/theLog.txt'
        self.log_file = None
        self.solution_file_path = 'solutions/theSolution.txt'
        self.config_parser = None
        self.curr_run = 0


    def set_up_from_config_file(self, config_file_path, problem_file_path):
        """
        Set up an experiment given a configuration file path and
        problem file path.

        Return True if successful, False if not.
        """
        try:
            self.config_parser = configparser.ConfigParser()
            self.config_parser.read(config_file_path)

            try:
                self.random_seed = self.config_parser.getint('basic_options', 'random_seed')
                print('config: random_seed =', self.random_seed)
            except:
                print('config: random_seed not specified; using system time')
                self.random_seed = int(time.time() * 1000.0)
            random.seed(self.random_seed)

            try:
                self.search_strategy_name = self.config_parser.get('basic_options', 'search_strategy')
                self.search_strategy_name = self.search_strategy_name.lower()
                print('config: search_strategy =', self.search_strategy_name)
            except:
                print('config: search_strategy not properly specified; using', self.search_strategy_name)

            try:
                self.num_runs_per_experiment = self.config_parser.getint('basic_options', 'num_runs_per_experiment')
                print('config: num_runs_per_experiment =', self.num_runs_per_experiment)
            except:
                print('config: num_runs_per_experiment not properly specified; using', self.num_runs_per_experiment)

            try:
                self.num_fitness_evals_per_run = self.config_parser.getint('basic_options',
                                                                           'num_fitness_evals_per_run')
                print('config: num_fitness_evals_per_run =', self.num_fitness_evals_per_run)
            except:
                print('config: num_fitness_evals_per_run not properly specified; using',
                      self.num_fitness_evals_per_run)

            try:
                self.log_file_path = self.config_parser.get('basic_options', 'log_file_path')
                print('config: log_file_path =', self.log_file_path)
            except:
                print('config: log_file_path not properly specified; using', self.log_file_path)

            try:
                self.solution_file_path = self.config_parser.get('basic_options', 'solution_file_path')
                print('config: solution_file_path =', self.solution_file_path)
            except:
                print('config: solution_file_path not properly specified; using', self.solution_file_path)

            # Dump parms to log file
            try:
                self.log_file = open(self.log_file_path, 'w')

                self.log_file.write('Result Log\n\n')
                self.log_file.write('problem file path: ' + problem_file_path + '\n')
                self.log_file.write('solution file path: ' + self.solution_file_path + '\n')
                self.log_file.write('random seed: ' + str(self.random_seed) + '\n')
                self.log_file.write('search strategy: ' + self.search_strategy_name + '\n')
                self.log_file.write('number of runs per experiment: ' + str(self.num_runs_per_experiment) + '\n')
                self.log_file.write('num_fitness_evals_per_run: '
                                    + str(self.num_fitness_evals_per_run) + '\n')
            except:
                print('config: problem with log file', self.log_file_path)
                traceback.print_exc()
                return False

            return True

        except:
            traceback.print_exc()
            return False


    def run_experiment(self, puzzle_state):
        """
        Run the experiment defined by the member variables contained in this
        experiment instance on the provided puzzle state.
        """

        if (self.search_strategy_name == 'random'):
            self.search_strategy = RandomStrategy(self)
        elif (self.search_strategy_name == 'ea'):
            self.search_strategy = EAStrategy(self)
        elif (self.search_strategy_name == 'moea'):
            self.search_strategy = MOEAStrategy(self)
        else:
            print('Error: unknown strategy \"',self.search_strategy_name,'\" -- cannot proceed')
            sys.exit(1)

        max_fitness_overall = float('-inf')
        best_pareto_front_overall = None
        best_puzzle_state = puzzle_state

        print('\nPuzzle has', puzzle_state.rows * puzzle_state.cols, 'cells;', puzzle_state.black_cells, 'black cells; max possible score', puzzle_state.max_possible_fitness)
        puzzle_state.print_puzzle_ANSI_text()

        # For each run...
        for curr_run in range(1, self.num_runs_per_experiment + 1):
            self.curr_run = curr_run
            print('\nRun', curr_run)
            self.log_file.write('\nRun ' + str(curr_run) + '\n')

            # Run the strategy and find the best solution for the current run
            if (self.search_strategy_name == 'moea'):
                best_pareto_front_this_run = self.search_strategy.execute_one_run(puzzle_state, self)
                # If this is the best Pareto front found in any run, save a copy
                # of the front as our candidate for best overall front.
                if (self.search_strategy.is_better_front(best_pareto_front_this_run,
                                                         best_pareto_front_overall,
                                                         self.search_strategy.fitness_function == '4_obj')):
                    best_pareto_front_overall = best_pareto_front_this_run
            else:
                best_puzzle_state_this_run = self.search_strategy.execute_one_run(puzzle_state, self)
                # If this is the highest score found in any run, save a copy
                # of the puzzle state as our candidate for best overall solution.
                if (best_puzzle_state_this_run.fitness > max_fitness_overall):
                    max_fitness_overall = best_puzzle_state_this_run.fitness
                    best_puzzle_state = copy.deepcopy(best_puzzle_state_this_run)

        # Close out the log file
        if (not(self.log_file is None)):
            self.log_file.close()

        # Write the solution file.
        print()
        try:
            with open(self.solution_file_path, 'w') as writer:
                print('Writing solution:')
                writer.write(best_puzzle_state.get_puzzle_string_for_solution_file())
                if (self.search_strategy_name == 'moea'):
                    print('Size of Best Pareto Front overall:', len(best_pareto_front_overall))
                    for genotype in best_pareto_front_overall:
                        best_puzzle_state.decode_state_from(genotype.encoding)
                        best_puzzle_state.calculate_fitness_mo()
                        writer.write(best_puzzle_state.get_solution_string_mo(self.search_strategy.fitness_function == '4_obj'))
                else:
                    writer.write(best_puzzle_state.get_solution_string())
                    print('fitness:', best_puzzle_state.strict_fitness,
                          'valid:', best_puzzle_state.is_valid)
                    best_puzzle_state.print_puzzle_ANSI_text()
        except:
            print('Could not open or write to solution file',
                  self.solution_file_path, '--aborting')
            sys.exit(1)

        # Q&D write some data to plot
        if (self.search_strategy_name == 'moea'):
            try:
                basename = self.solution_file_path.split('/')[1]
                with open('data/' + basename, 'w') as writer:
                    for genotype in best_pareto_front_overall:
                        best_puzzle_state.decode_state_from(genotype.encoding)
                        best_puzzle_state.calculate_fitness_mo()
                        writer.write(str(best_puzzle_state.fitness_mo[0]) + '\t')
                        writer.write(str(best_puzzle_state.fitness_mo[1]) + '\t')
                        writer.write(str(best_puzzle_state.fitness_mo[2]) + '\t')
                        writer.write(str(best_puzzle_state.fitness_mo[3]) + '\n')
            except:
                print('Could not open or write to data/plotit.txtsolution file -- aborting')
                sys.exit(1)





