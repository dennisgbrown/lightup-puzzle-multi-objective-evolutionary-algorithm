# -*- coding: utf-8 -*-
import numpy

# Give names to cell state values to increase code readability
OPEN_CELL = -1
BLACK_ZERO = 0
BLACK_ONE = 1
BLACK_TWO = 2
BLACK_THREE = 3
BLACK_FOUR = 4
BLACK_ANY = 5
BULB = 6
LIT = 7
BULB_FORBIDDEN = 8


class PuzzleState:
    """
    This class holds the puzzle state:
    - a matrix containing the board state
    - number of rows, columns, and black cells
    - fitness value of the puzzle
    along with methods to read and write to file, calculate fitness,
    and print the board to the screen (plus all their helper methods).

    PuzzleState is the Phenotype of an individual in an EA
    """
    def __init__(self):
        self.puzzle = None
        self.rows = 0
        self.cols = 0
        self.black_cells = 0
        self.max_possible_fitness = 0

        # Strict fitness -- 0 if any violations, otherwise number of cells lit
        self.strict_fitness = 0

        # Working fitness for various single-fitness methods
        self.fitness = 0

        # Working fitness for MOEA operations
        self.fitness_mo = 0


    def read_puzzle_from_file(self, filename):
        """
        Load the puzzle state from a file in the format specified in
        the assignment.

        The internal state of the puzzle is stored in 0-indexed
        rows and columns. Translate to 1-indexed columns and rows
        as the file is read.

        Return True if successful, False if not.
        """
        self.rows = 0
        self.cols = 0
        self.black_cells = 0

        try:
            with open(filename, 'r') as reader:
                # Try to read number of columns and rows
                try:
                    self.cols = int(reader.readline())
                    self.rows = int(reader.readline())
                except:
                    print("Could not parse X or Y size--aborting")
                    return False

                # Leverage numpy's array features to create board state
                self.puzzle = numpy.zeros((self.rows, self.cols))
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.puzzle[i][j] = OPEN_CELL

                # Read the black cell locations line-by-line
                for line in reader:
                    black_cell_list = list(line.split(" "))

                    # Break each line up into x lcoation, y lcoation, cell value
                    x = int(black_cell_list[0])
                    if ((x < 1) or (x > self.cols)):
                        print("x value out of range:", x)
                        return False

                    y = int(black_cell_list[1])
                    if ((y < 1) or (y > self.rows)):
                        print("y value out of range:", y)
                        return False

                    val = int(black_cell_list[2])

                    # And set the value into the array
                    self.puzzle[y - 1][x - 1] = val
                    self.black_cells += 1

                self.max_possible_fitness = (self.rows * self.cols) - self.black_cells
                return True

        except:
            print("Could not open or parse puzzle file", filename, "--aborting")
            return False


    def get_puzzle_string_for_solution_file(self):
        """
        Return a string consisting of the puzzle board for the solution file.

        The internal state of the puzzle is stored in 0-indexed
        rows and columns. Translate to 1-indexed columns and rows
        as the file is written.
        """
        return_string = str(self.cols) + '\n' + str(self.rows) + '\n'

        # Write the black cell locations and values
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] >= BLACK_ZERO) and (self.puzzle[i][j] <= BLACK_ANY)):
                    return_string += str(j + 1) + ' ' + str(i + 1) + ' ' + str(int(self.puzzle[i][j])) + '\n'

        return return_string


    def get_solution_string(self):
        """
        Return a string consisting of the solution
        to be written to a solution file.
        """

        return_string = str(self.strict_fitness) + '\n'

        # Write the bulb locations
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.puzzle[i][j] == BULB):
                    return_string += str(j + 1) + ' ' + str(i + 1) + '\n'

        return return_string


    def get_solution_string_mo(self, four_obj):
        """
        Return a string consisting of the solution (multi-objective)
        to be written to a solution file.
        """

        return_string = str(self.fitness_mo[0]) + '\t'
        return_string += str(self.fitness_mo[1]) + '\t'
        return_string += str(self.fitness_mo[2])
        if (four_obj):
            return_string += '\t' + str(self.fitness_mo[3])
        return_string += '\n'

        # Write the bulb locations
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.puzzle[i][j] == BULB):
                    return_string += str(j + 1) + ' ' + str(i + 1) + '\n'

        return return_string


    def get_encoding(self):
        """
        Encoding is naive binary: Ignoring black cells, walk each cell of each
        row of board state adding a digit to encoding for each cell where
        bulb = 1 and other = 0
        """
        encoding = []
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] < BLACK_ZERO)
                    or (self.puzzle[i][j] > BLACK_ANY)):
                    if (self.puzzle[i][j] == BULB):
                        encoding.append(1)
                    else:
                        encoding.append(0)
        return encoding


    def decode_state_from(self, encoding):
        """
        Decode from naive binary: Ignoring black cells, walk each cell of each
        row of board state inserting bulb or open cell based on next digit
        of encording, where 1 = bulb and 0 = open cell
        """
        curr_bit = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] < BLACK_ZERO)
                    or (self.puzzle[i][j] > BLACK_ANY)):
                    if (encoding[curr_bit] == 1):
                        self.puzzle[i][j] = BULB
                    else:
                        self.puzzle[i][j] = OPEN_CELL
                    curr_bit += 1


    def check_bulb_cell_and_set_lit_cells(self, i, j):
        """
        Check if location i,j is a valid location for a bulb.
        Assume valid until we find a condition that makes it invalid
        (the bulb is shining on another bulb).

        This method has side effects: if a cell is light by a bulb,
        that is stored in the board state.

        Return True or False.
        """
        is_valid = True

        # Check cells to the left of the bulb.
        if (j > 0):
            for check_j in range(j - 1, -1, -1):
                # If we hit a black cell, we can stop checking.
                if ((self.puzzle[i][check_j] >= BLACK_ZERO)
                    and (self.puzzle[i][check_j] <= BLACK_ANY)):
                    break
                # If we hit a bulb cell, this location is invalid.
                if (self.puzzle[i][check_j] == BULB):
                    is_valid = False
                # Otherwise we can light up this cell if it's empty
                elif (self.puzzle[i][check_j] == OPEN_CELL):
                    self.puzzle[i][check_j] = LIT

        # Check cells to the right of the bulb.
        if (j < (self.cols - 1)):
            for check_j in range(j + 1, self.cols, 1):
                # If we hit a black cell, we can stop checking.
                if ((self.puzzle[i][check_j] >= BLACK_ZERO)
                    and (self.puzzle[i][check_j] <= BLACK_ANY)):
                    break
                # If we hit a bulb cell, this location is invalid.
                if (self.puzzle[i][check_j] == BULB):
                    is_valid = False
                # Otherwise we can light up this cell if it's empty
                elif (self.puzzle[i][check_j] == OPEN_CELL):
                    self.puzzle[i][check_j] = LIT

        # Check cells below the bulb.
        if (i > 0):
            for check_i in range(i - 1, -1, -1):
                # If we hit a black cell, we can stop checking.
                if ((self.puzzle[check_i][j] >= BLACK_ZERO)
                    and (self.puzzle[check_i][j] <= BLACK_ANY)):
                    break
                # If we hit a bulb cell, this location is invalid.
                if (self.puzzle[check_i][j] == BULB):
                    is_valid = False
                # Otherwise we can light up this cell if it's empty
                elif (self.puzzle[check_i][j] == OPEN_CELL):
                    self.puzzle[check_i][j] = LIT

        # Check cells above the bulb.
        if (i < (self.rows - 1)):
            for check_i in range(i + 1, self.rows, 1):
                # If we hit a black cell, we can stop checking.
                if ((self.puzzle[check_i][j] >= BLACK_ZERO)
                    and (self.puzzle[check_i][j] <= BLACK_ANY)):
                    break
                # If we hit a bulb cell, this location is invalid.
                if (self.puzzle[check_i][j] == BULB):
                    is_valid = False
                # Otherwise we can light up this cell if it's empty
                elif (self.puzzle[check_i][j] == OPEN_CELL):
                    self.puzzle[check_i][j] = LIT

        return is_valid


    def check_black_numbered_cell(self, i, j):
        """
        Check number of bulb neighbors for black cell at location i,j.

        Return a tuple of (num required, num present)
        """
        num_light_neighbors = 0

        # Check left neighbor
        if ((j > 0) and (self.puzzle[i][j - 1] == BULB)):
            num_light_neighbors += 1
        # Check right neighbor
        if ((j < (self.cols - 1)) and (self.puzzle[i][j + 1] == BULB)):
            num_light_neighbors += 1
        # Check lower neighbor
        if ((i > 0) and (self.puzzle[i - 1][j] == BULB)):
            num_light_neighbors += 1
        # Check upper neighbor
        if ((i < (self.rows - 1)) and (self.puzzle[i + 1][j] == BULB)):
            num_light_neighbors += 1

        return (self.puzzle[i][j], num_light_neighbors)


    def place_unique_bulbs(self):
        """
        Place bulbs in empty cells around black cells when there is only
        one way to do so.
        """

        # Keep trying to add bulbs until we go an iteration without placing any
        keep_trying = True
        while (keep_trying):
            keep_trying = False

            # First mark forbidden areas around "0" black cells
            for i in range(self.rows):
                for j in range(self.cols):
                    if (self.puzzle[i][j] == BLACK_ZERO):
                        # Check left neighbor
                        if ((j > 0)
                            and (self.puzzle[i][j - 1] == OPEN_CELL)):
                            self.puzzle[i][j - 1] = BULB_FORBIDDEN
                        # Check right neighbor
                        if ((j < (self.cols - 1))
                            and (self.puzzle[i][j + 1] == OPEN_CELL)):
                            self.puzzle[i][j + 1] = BULB_FORBIDDEN
                        # Check lower neighbor
                        if ((i > 0)
                            and (self.puzzle[i - 1][j] == OPEN_CELL)):
                            self.puzzle[i - 1][j] = BULB_FORBIDDEN
                        # Check upper neighbor
                        if ((i < (self.rows - 1))
                            and (self.puzzle[i + 1][j] == OPEN_CELL)):
                            self.puzzle[i + 1][j] = BULB_FORBIDDEN

            # Then check each black cell for open spaces; compare to
            # number of bulbs allowed, and if it's a match, fill the bulbs in.
            for i in range(self.rows):
                for j in range(self.cols):
                    if ((self.puzzle[i][j] >= BLACK_ONE)
                        and (self.puzzle[i][j] <= BLACK_FOUR)):
                        candidate_cells = []
                        # Check left neighbor
                        if ((j > 0)
                            and ((self.puzzle[i][j - 1] == OPEN_CELL)
                                 or (self.puzzle[i][j - 1] == BULB))):
                            candidate_cells.append((i, j - 1))
                        # Check right neighbor
                        if ((j < (self.cols - 1))
                            and ((self.puzzle[i][j + 1] == OPEN_CELL)
                                 or (self.puzzle[i][j + 1] == BULB))):
                            candidate_cells.append((i, j + 1))
                        # Check lower neighbor
                        if ((i > 0)
                            and ((self.puzzle[i - 1][j] == OPEN_CELL)
                                 or (self.puzzle[i - 1][j] == BULB))):
                            candidate_cells.append((i - 1, j))
                        # Check upper neighbor
                        if ((i < (self.rows - 1))
                            and ((self.puzzle[i + 1][j] == OPEN_CELL)
                                 or (self.puzzle[i + 1][j] == BULB))):
                            candidate_cells.append((i + 1, j))

                        if (len(candidate_cells) ==
                            ((self.puzzle[i][j] - BLACK_ONE) + 1)):
                            for new_bulb in candidate_cells:
                                if (self.puzzle[new_bulb[0]][new_bulb[1]] != BULB):
                                    self.puzzle[new_bulb[0]][new_bulb[1]] = BULB
                                    self.check_bulb_cell_and_set_lit_cells(new_bulb[0], new_bulb[1])
                                    keep_trying = True

        self.clean()


    def reset_cell_type(self, cell_type):
        """
        For a given cell type, reset all matching cells in the board state.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.puzzle[i][j] == cell_type):
                    self.puzzle[i][j] = OPEN_CELL


    def reset(self):
        """
        Reset all lit cells, bulb cells, bulb forbidden cells, and fitness value.
        """
        self.reset_cell_type(LIT)
        self.reset_cell_type(BULB)
        self.reset_cell_type(BULB_FORBIDDEN)
        self.fitness = 0
        self.is_valid = False


    def clean(self):
        """
        Reset all lit cells and bulb forbidden cells.
        """
        self.reset_cell_type(LIT)
        self.reset_cell_type(BULB_FORBIDDEN)


    def reset_to(self, puzzle_state):
        """
        Reset the state of this puzzle to match that provided.
        Just the board state.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.puzzle[i][j] = puzzle_state.puzzle[i][j]


    def count_cell_type(self, cell_type):
        """
        For a given cell type, return how many are in the puzzle state.
        """
        count = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.puzzle[i][j] == cell_type):
                    count += 1
        return count


    def calculate_fitness(self, enforce_black_cells, with_penalty, c_p):
        """
        Return the fitness value for the current board state.

        enforce_black_cells = True means:
        - black cell violations mean self.strict_fitness = 0
        - black cell violations increase the penalty for penalized fitness

        enforce_black_cells = False means:
        - black cell violations have no effect on self.strict_fitness
        - black cell violations have no effect on penalty for penalized fitness

        with_penalty = True means:
        the working fitness (self.fitness) is (unrestricted fitness - penalty)

        with_penalty = False means:
        the working fitness (self.fitness) is same as self.strict_fitness

        enforce_black_cells = False and with_penalty = True
        means black cell violations are not included in penalty, only
        bulb violations (this combo is not really used)

        c_p is the penalty coefficient

        Returns working fitness value. Sets other values in the puzzle state.
        """
        # Clean up the puzzle state before proceeding
        self.clean()

        # Check for bulb shining violations and set lit cells
        bulb_cell_violations = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] == BULB)
                    and (not self.check_bulb_cell_and_set_lit_cells(i, j))):
                    bulb_cell_violations += 1

        # Calculate unrestricted fitness (ignore all violations)
        unrestricted_fitness = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] == BULB)
                    or (self.puzzle[i][j] == LIT)):
                        unrestricted_fitness += 1

        # clean up side effects of setting lit cells
        self.clean()

        # Check for black cell violations
        black_cell_violations = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] >= BLACK_ZERO)
                    and (self.puzzle[i][j] <= BLACK_FOUR)):
                    (required, actual) = self.check_black_numbered_cell(i, j);
                    if (required != actual):
                        black_cell_violations += abs(required - actual)

        # Set strict fitness
        self.strict_fitness = 0
        if (enforce_black_cells):
            if ((black_cell_violations == 0) and (bulb_cell_violations == 0)):
                self.strict_fitness = unrestricted_fitness
        else:
            if (bulb_cell_violations == 0):
                self.strict_fitness = unrestricted_fitness

        # Set working fitness
        if (with_penalty):
            if (enforce_black_cells):
                self.fitness = unrestricted_fitness - (c_p * (bulb_cell_violations + black_cell_violations))
            else:
                self.fitness = unrestricted_fitness - (c_p * bulb_cell_violations)
        else:
            self.fitness = self.strict_fitness

        self.is_valid = (self.strict_fitness > 0)

        return self.fitness


    def calculate_fitness_basic(self, enforce_black_cells):
        return self.calculate_fitness(enforce_black_cells, False, 0.0)


    def calculate_fitness_mo(self):
        """
        Calculate multiple sub-fitnesses
        """
        # Clean up the puzzle state before proceeding
        self.clean()

        # Check for bulb shining violations and set lit cells
        bulb_cell_violations = 0
        num_bulbs = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if (self.puzzle[i][j] == BULB):
                    num_bulbs -= 1
                    if (not self.check_bulb_cell_and_set_lit_cells(i, j)):
                        bulb_cell_violations -= 1

        # Number of cells lit up
        cells_lit = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] == BULB)
                    or (self.puzzle[i][j] == LIT)):
                        cells_lit += 1

        # clean up side effects of setting lit cells
        self.clean()

        # Check for black cell violations
        black_cell_violations = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if ((self.puzzle[i][j] >= BLACK_ZERO)
                    and (self.puzzle[i][j] <= BLACK_FOUR)):
                    (required, actual) = self.check_black_numbered_cell(i, j);
                    if (required != actual):
                        black_cell_violations -= abs(required - actual)

        self.is_valid = (bulb_cell_violations == black_cell_violations == 0)

        self.fitness_mo = (cells_lit, bulb_cell_violations,
                           black_cell_violations, num_bulbs)

        return self.fitness_mo


    def print_puzzle_values(self):
        """
        Print the board state using the built-in list printer.

        Board state is stored upside down compared to how it
        should be displayed; printing code corrects for that.
        """
        for i in range(self.rows):
            print(self.puzzle[self.rows - i - 1])


    @staticmethod
    def get_ASCII_string_for_cell_value(cell_value):
        """
        Returns a character based on cell type.
        """
        if (cell_value == OPEN_CELL):
            return(' ')
        if ((cell_value >= BLACK_ZERO) and (cell_value <= BLACK_FOUR)):
            return(str(int(cell_value)))
        if (cell_value == BLACK_ANY):
            return('W')  # W for whatever
        if (cell_value == BULB):
            return('O')
        if (cell_value == BULB_FORBIDDEN):
            return('X')
        if (cell_value == LIT):
            return('-')
        return(' ')


    @staticmethod
    def get_ASCII_border_block():
        """
        Returns a value to mark the edges of the board.
        """
        return('#')


    def print_puzzle_ASCII_text(self):
        """
        Print the board state using meaningful ASCII characters.

        Board state is stored upside down compared to how it
        should be displayed; printing code corrects for that.
        """
        row_string = ''
        for i in range(self.cols + 2):
            row_string += self.get_ASCII_border_block()
        print(row_string)

        for i in range(self.rows):
            row_string = ''
            for j in range(self.cols):
                row_string += self.get_ASCII_string_for_cell_value(self.puzzle[self.rows - i - 1][j])
            print(self.get_ASCII_border_block() + row_string + self.get_ASCII_border_block())

        row_string = ''
        for i in range(self.cols + 2):
            row_string += self.get_ASCII_border_block()
        print(row_string)


    @staticmethod
    def get_ANSI_string_for_cell_value(cell_value):
        """
        Returns a colored block based on cell type.
        """
        if (cell_value == OPEN_CELL):
            return('\033[0;30;47m \033[0m')
        if ((cell_value >= BLACK_ZERO) and (cell_value <= BLACK_FOUR)):
            #return('\033[1;37;40m' + str(int(cell_value)) + '\033[0m')
            return(str(int(cell_value)))
        if (cell_value == BLACK_ANY):
            #return('\033[1;37;40m \033[0m')
            return(' ')
        if (cell_value == BULB):
            return('\033[0;30;47mO\033[0m')
        if (cell_value == BULB_FORBIDDEN):
            return('\033[0;30;47mX\033[0m')
        if (cell_value == LIT):
            return('\033[0;30;47m-\033[0m')
        return(' ')


    @staticmethod
    def get_ANSI_border_block():
        """
        Returns a colored solid block.
        """
        return('\033[0;30;43m \033[0m')


    def print_puzzle_ANSI_text(self):
        """
        Print the board state using meaningful and colorful ANSI characters.

        Board state is stored upside down compared to how it
        should be displayed; printing code corrects for that.
        """
        row_string = ''
        for i in range(self.cols + 2):
            row_string += self.get_ANSI_border_block()
        print(row_string)

        for i in range(self.rows):
            row_string = ''
            for j in range(self.cols):
                row_string += self.get_ANSI_string_for_cell_value(self.puzzle[self.rows - i - 1][j])
            print(self.get_ANSI_border_block() + row_string + self.get_ANSI_border_block())

        row_string = ''
        for i in range(self.cols + 2):
            row_string += self.get_ANSI_border_block()
        print(row_string)


