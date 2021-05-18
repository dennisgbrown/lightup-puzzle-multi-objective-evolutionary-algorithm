# README for COMP 6666 Fall 2020 Assignment 1d
Dennis Brown / dgb0028@auburn.edu / 11 OCT 2020

## Overview 

Multi-objective evolutionary algorithm to play the [Light Up puzzle game](https://en.wikipedia.org/wiki/Light_Up_(puzzle)).

See the [project report](https://github.com/dennisgbrown/comp6666-lightup-moea/blob/master/assignment1d_report.pdf) for more context.

## Usage

Code was written in Python 3 and is all in the "code" folder.

Run the assignment code by using run.sh as specified in the assignment instructions (see end of this document).
```
./run.sh problem_filepath
```
```
./run.sh problem_filepath optional_config_filepath
```

The working directory is assumed to be the main assignment folder containing run.sh (i.e. one level above code / logs / problems / solutions folders).

Config files are in the configs subfolder. If no config file is specified, "default.config" is used and you're stuck with whatever is in it.

Solution and log files go into their respective subfolders; file names have the same root filename as the config files.

There are many config files as follows:
* GREEN 3:
    * d1/d2-100-50-fps-plus-trunc.config: for provided problem instances, chosen baseline MOEA configuration -- full description in the report
    * d1/d2-100-50-fps-plus-rand.config: baseline but with Uniform Random survivor selection
    * d1/d2-100-50-fps-plus-fps.config: baseline but with FPS survivor selection
    * d1/d2-100-50-k10-plus-trunc.config: baseline but with k-Tournament (10) parent selection
    * d1/d2-100-100-fps-comma-trunc.config: baseline but with comma survival strategy (mu=100, lamda=100)
    * d1/d2-100-100-fps-plus-trunc-noval.config: baseline but validity enforced initializaton disabled
* YELLOW 1:
    * d1/d2-100-50-fps-plus-trunc-share10.config: baseline but with Fitness Sharing (sigma=10)
    * d1/d2-100-50-fps-plus-trunc-share10.config: baseline but with Fitness Sharing (sigma=50)
* YELLOW 2:
    * d1/d2-100-50-fps-plus-trunc-crowd.config: baseline but with Crowding
* RED 1:
    * d1/d2-100-50-fps-plus-trunc-4obj.config: baseline but with 4 objectives instead of 3

Omitting the random seed from the config file results in initializing the random seed to an integer version of system time that is also written to the log file.

Malformed inputs generally cause the program to report an error and halt. Default values are employed where applicable, somewhat arbitrarily. User is highly encouraged to use command line and config file properly.

This code has been tested on the Tux network.

## Report

See file **assignment1d_report.pdf**

## Architecture

Execution kicks off in *start.py*, which parses command line arguments and sets up PuzzleState and Experiment instances.

The *PuzzleState* class contains the state of the LightUp/Akari puzzle, reads puzzle files as specified in the assignment, calculates fitness, checks and resets light positions, checks black cell constraints, and dumps board state to stdout.

&rarr; PuzzleState is mutable. Its operations have side effects on its state.

&rarr; The board state is stored in a numpy 2D array indexed by row, col starting at 0. Coordinates are translated as needed when reading/writing board state.

The *Experiment* class contains the experiment parameters read from the config file, sets up the experiment, and executes the runs and evaluations of the experiment using a specified *Strategy*.

The *Strategy* class is the base class for solution strategies. It specifies
initialization and execution (for one run) methods.

The *RandomStrategy* class executes one run of the experiment given
a puzzle and parameters in an Experiment instance. It generates
solutions using a naive and direct implementation of the assignment's requirement
that it "generates uniform random placement for a uniform random number of
bulbs between 1 and the number of white cells, to find the valid solution which
maximizes the number of white cells which are lit up where a cell containing a
bulb is also considered lit."

The *EAStrategy* class executes one run of the experiment given
a puzzle and parameters in an Experiment instance. It generates
solutions using the steady state (mu + lambda) or generational (mu, lambda)
strategies, with the original/strict fitness function or a
constraint-satisfaction (penalized) fitness function. It has many
parameters configurable in the config file per the assignment instructions.
It supports all requirements of assignments 1b and 1c and is not used in
assignment 1d. It is largely unchanged for assignment 1d except to address
some comments from feedback on assignment 1c, in case the grader is looking
for those.

The *MOEAStrategy* class executes one run of the experiment given
a puzzle and parameters in an Experiment instance. It generates
solutions using a Multi-Objective EA described in the accompanying report
for assignment 1d. It is a duplication and adaptation of "EAStrategy" so
the two classes would benefit from refactoring, but that is last on my
priority list for this assignment. Some things to point out:
* The code maintains 4 objectives for the multi-objective fitness values
(to support RED 1), but it only actually uses the first 3 objectives
(GREEN 1-3, YELLOW 1-2) unless the fitness_function configuration value is set to "4_obj."
* In my NSGA-II implementation, the best level is 0 and the remaining levels
are increasingly negative integers.

The *runPlotterMo*, *runPlotter2Mo*, *runPlotter2MoDiv*, and *solutionPlotter*
modules are just
helpful utilities for me to make plots. At this point the names are
nonsensical and the code quality is atrocious. Please ignore them.


------------

# Original README contents:
#################################
#	Problem Instances	#
#################################

Light up puzzle instances d1.lup and d2.lup should be used for all comparisons. Straightforward puzzle use! Woo!

#################################
#	Coding Standards	#
#################################

You are free to use any of the following programming languages for your submission :
	- Python 3
	- C++
	- C#
	- Java

NOTE : Sloppy, undocumented, or otherwise unreadable code will be penalized for not following good coding standards (as laid out in the grading rubric on the course website) : https://www.eng.auburn.edu/~drt0015/coding.html

#################################
#	Submission Rules	#
#################################

Included in your repository is a script named ”finalize.sh”, which you will use to indicate which version of your code is the one to be graded. When you are ready to submit your final version, run the command ./finalize.sh or ./finalize.sh -language_flag from your local Git directory, then commit and push your code. Running the finalize script without a language flag will cause the script to run an interactive prompt where you may enter your programming language. Alternatively, you can pass a -j flag when running the finalize script to indicate that you are submitting in Java (i.e. ./finalize.sh -j). The flag -j indicates Java, -cpp indicates C++, -cs indicates C#, and -p indicates Python 3. This script also has an interactive prompt where you will enter your Auburn username so the graders can identify you. The finalize script will create a text file, readyToSubmit.txt, that is populated with information in a known format for grading purposes. You may commit and push as much as you want, but your submission will be confirmed as ”final” if ”readyToSubmit.txt” exists and is populated with the text generated by ”finalize.sh” at 10:00pm on the due date. If you do not plan to submit before the deadline, then you should NOT run the ”finalize.sh” script until your final submission is ready. If you accidentally run ”finalize.sh” before you are ready to submit, do not commit or push your repo and delete ”readyToSubmit.txt.” Once your final submission is ready, run ”finalize.sh”, commit and push your code, and do not make any further changes to it

Late submissions will be penalized 5% for the first 24 hour period and an additional 10% for every 24 hour period thereafter.

#################################
#       Compiling & Running	#
#################################

Your final submission must include the script "run.sh" which should compile and run your code.

Your script should run on a standard linux machines with the following commands :
```
./run.sh problem_filepath
```
```
./run.sh problem_filepath optional_config_filepath
```
