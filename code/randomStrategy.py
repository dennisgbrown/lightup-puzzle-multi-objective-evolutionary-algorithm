# -*- coding: utf-8 -*-
import random
import copy
import sys

sys.path.append('code')
import puzzleState
from strategy import Strategy

class RandomStrategy(Strategy):
    """
    Class implementing the random strategy.
    """

    def __init__(self, experiment):
        pass


    def execute_one_run(self, puzzle_state, experiment):
        puzzle_state.reset()
        max_fitness_this_run = -1
        best_puzzle_state_this_run = None
        total_fitness = 0

        #... run the specified number of evaluations.
        for curr_eval in range(1, experiment.num_fitness_evals_per_run + 1):
            puzzle_state.reset()

            self.generate_solution_candidate(puzzle_state)

            # Provide status message every nth evaluation.
            if ((curr_eval % 1000) == 0):
                print('\r', curr_eval, 'evals', end =" ")

            fitness = puzzle_state.calculate_fitness_basic(experiment.enforce_black_cell_constraint)
            total_fitness += fitness

            # If this is the highest score in this run, add the result
            # to the log file and print some useful diagnostic information.
            if (fitness > max_fitness_this_run):
                max_fitness_this_run = fitness
                max_fitness_this_run = fitness
                best_puzzle_state_this_run = copy.deepcopy(puzzle_state)

        print()
        print('average: ' + str(total_fitness / curr_eval))
        print('best: ' + str(max_fitness_this_run))
        experiment.log_file.write(str(curr_eval) + '\t'
                                  + str(total_fitness / curr_eval) + '\t'
                                  + str(max_fitness_this_run) + '\t'
                                  + '\n')

        return best_puzzle_state_this_run


    def generate_solution_candidate(self, puzzle_state):
        """
        Generate a single solution candidate within the provided PuzzleState
        instance. It has side effects on the puzzle state. It is a naive and
        direct implementation of the assignment's requirement that it
        "generates uniform random placement for a uniform random number of bulbs
        between 1 and the number of white cells, to find the valid solution
        which maximizes the number of white cells which are lit up where a cell
        containing a bulb is also considered lit."
        """

        # Pick a random number of bulbs to place
        num_bulbs = int(random.randint(1, puzzle_state.max_possible_fitness))

        # For each bulb, loop forever until we pick an open cell.
        for bulb in range(num_bulbs):
            while True:
                i = int(random.randint(0, puzzle_state.rows - 1))
                j = int(random.randint(0, puzzle_state.cols - 1))
                if (puzzle_state.puzzle[i][j] == puzzleState.OPEN_CELL):
                    puzzle_state.puzzle[i][j] = puzzleState.BULB
                    break


