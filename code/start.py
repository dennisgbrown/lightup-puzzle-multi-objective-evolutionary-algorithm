# -*- coding: utf-8 -*-
import sys

sys.path.append('code')
from puzzleState import PuzzleState
from experiment import Experiment

def main():
    """
    Parse command line arguments, create a puzzle state, and
    run the experiment.
    """
    problem_file_path = 'problems/bc1.lup'
    config_file_path = 'configs/default.config'

    if (len(sys.argv) > 1):
        print(f'The problem file passed is: {sys.argv[1]}')
        problem_file_path = sys.argv[1]
    else:
        print('No problem file specified -- using', problem_file_path)

    if (len(sys.argv) > 2):
        print(f'The config file passed is: {sys.argv[2]}')
        config_file_path = sys.argv[2]
    else:
        print('No config file specified -- using', config_file_path)

    puzzle_state = PuzzleState()
    if (not puzzle_state.read_puzzle_from_file(problem_file_path)):
        print('Error opening or parsing puzzle file -- cannot proceed')
        sys.exit(1)

    experiment = Experiment()
    if (not experiment.set_up_from_config_file(config_file_path, problem_file_path)):
        print('Error opening or parsing config file -- cannot proceed')
        sys.exit(1)

    experiment.run_experiment(puzzle_state)

    print('\n---Experiment concluded---')

if __name__ == '__main__':
    main()


