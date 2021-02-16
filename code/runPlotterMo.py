# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import traceback

"""
Read a log file plot best vs. average with "error bars" to show min-max range

Do it for multiple objectives.
"""

fileroot = 'd1-100-50-fps-plus-trunc-share50'

filename = 'logs/' + fileroot + '.txt'

four_obj = False
#four_obj = True

curr_run = -1
lineno = 0

run_evals = []
run_average_lists = []
run_best_lists = []
run_average_lists.append([])
run_average_lists.append([])
run_average_lists.append([])
run_best_lists.append([])
run_best_lists.append([])
run_best_lists.append([])
if (four_obj):
    run_average_lists.append([])
    run_best_lists.append([])

with open(filename, 'r') as reader:
   curr_line = reader.readline()
   while (curr_line):
        lineno += 1
        curr_line = curr_line.strip()

        if (curr_line.startswith('Run')):
            curr_run += 1
            curr_eval = 0

        elif ((len(curr_line) > 0) and (curr_run > -1)):
            try:
                data_list = list(curr_line.split("\t"))

                if (len(run_evals) < (curr_eval + 1)):
                    run_evals.append(int(data_list[0]))
                if (len(run_average_lists[0]) < (curr_eval + 1)):
                    run_average_lists[0].append([])
                if (len(run_average_lists[1]) < (curr_eval + 1)):
                    run_average_lists[1].append([])
                if (len(run_average_lists[2]) < (curr_eval + 1)):
                    run_average_lists[2].append([])
                if (four_obj and (len(run_average_lists[3]) < (curr_eval + 1))):
                    run_average_lists[3].append([])
                if (len(run_best_lists[0]) < (curr_eval + 1)):
                    run_best_lists[0].append([])
                if (len(run_best_lists[1]) < (curr_eval + 1)):
                    run_best_lists[1].append([])
                if (len(run_best_lists[2]) < (curr_eval + 1)):
                    run_best_lists[2].append([])
                if (four_obj and len(run_best_lists[3]) < (curr_eval + 1)):
                    run_best_lists[3].append([])

                run_average_lists[0][curr_eval].append(float(data_list[1]))
                run_best_lists[0][curr_eval].append(float(data_list[2]))
                run_average_lists[1][curr_eval].append(float(data_list[3]))
                run_best_lists[1][curr_eval].append(float(data_list[4]))
                run_average_lists[2][curr_eval].append(float(data_list[5]))
                run_best_lists[2][curr_eval].append(float(data_list[6]))
                if (four_obj):
                    run_average_lists[3][curr_eval].append(float(data_list[7]))
                    run_best_lists[3][curr_eval].append(float(data_list[8]))

                curr_eval += 1

            except:
                print('Problem in line ' + str(lineno) + ': |' + curr_line + '|')
                traceback.print_exc()
                pass

        curr_line = reader.readline()

run_evals = numpy.array(run_evals)

# I am not proud of the following code.

run_average_averages = []
run_average_averages.append(numpy.zeros(len(run_evals)))
run_average_averages.append(numpy.zeros(len(run_evals)))
run_average_averages.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_average_averages.append(numpy.zeros(len(run_evals)))

run_max_averages = []
run_max_averages.append(numpy.zeros(len(run_evals)))
run_max_averages.append(numpy.zeros(len(run_evals)))
run_max_averages.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_max_averages.append(numpy.zeros(len(run_evals)))

run_min_averages = []
run_min_averages.append(numpy.zeros(len(run_evals)))
run_min_averages.append(numpy.zeros(len(run_evals)))
run_min_averages.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_min_averages.append(numpy.zeros(len(run_evals)))

run_std_averages = []
run_std_averages.append(numpy.zeros(len(run_evals)))
run_std_averages.append(numpy.zeros(len(run_evals)))
run_std_averages.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_std_averages.append(numpy.zeros(len(run_evals)))

run_average_bests = []
run_average_bests.append(numpy.zeros(len(run_evals)))
run_average_bests.append(numpy.zeros(len(run_evals)))
run_average_bests.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_average_bests.append(numpy.zeros(len(run_evals)))

run_max_bests = []
run_max_bests.append(numpy.zeros(len(run_evals)))
run_max_bests.append(numpy.zeros(len(run_evals)))
run_max_bests.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_max_bests.append(numpy.zeros(len(run_evals)))

run_min_bests = []
run_min_bests.append(numpy.zeros(len(run_evals)))
run_min_bests.append(numpy.zeros(len(run_evals)))
run_min_bests.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_min_bests.append(numpy.zeros(len(run_evals)))

run_std_bests = []
run_std_bests.append(numpy.zeros(len(run_evals)))
run_std_bests.append(numpy.zeros(len(run_evals)))
run_std_bests.append(numpy.zeros(len(run_evals)))
if (four_obj):
    run_std_bests.append(numpy.zeros(len(run_evals)))

for i in range(len(run_evals)):
    run_average_averages[0][i] = numpy.mean(run_average_lists[0][i])
    run_average_averages[1][i] = numpy.mean(run_average_lists[1][i])
    run_average_averages[2][i] = numpy.mean(run_average_lists[2][i])
    if (four_obj):
        run_average_averages[3][i] = numpy.mean(run_average_lists[3][i])
    run_max_averages[0][i] = numpy.amax(run_average_lists[0][i])
    run_max_averages[1][i] = numpy.amax(run_average_lists[1][i])
    run_max_averages[2][i] = numpy.amax(run_average_lists[2][i])
    if (four_obj):
        run_max_averages[3][i] = numpy.amax(run_average_lists[3][i])
    run_min_averages[0][i] = numpy.amin(run_average_lists[0][i])
    run_min_averages[1][i] = numpy.amin(run_average_lists[1][i])
    run_min_averages[2][i] = numpy.amin(run_average_lists[1][i])
    if (four_obj):
        run_min_averages[3][i] = numpy.amin(run_average_lists[3][i])
    run_std_averages[0][i] = numpy.std(run_average_lists[0][i])
    run_std_averages[1][i] = numpy.std(run_average_lists[1][i])
    run_std_averages[2][i] = numpy.std(run_average_lists[2][i])
    if (four_obj):
        run_std_averages[3][i] = numpy.std(run_average_lists[3][i])
    run_average_bests[0][i] = numpy.mean(run_best_lists[0][i])
    run_average_bests[1][i] = numpy.mean(run_best_lists[1][i])
    run_average_bests[2][i] = numpy.mean(run_best_lists[2][i])
    if (four_obj):
        run_average_bests[3][i] = numpy.mean(run_best_lists[3][i])
    run_max_bests[0][i] = numpy.amax(run_best_lists[0][i])
    run_max_bests[1][i] = numpy.amax(run_best_lists[1][i])
    run_max_bests[2][i] = numpy.amax(run_best_lists[2][i])
    if (four_obj):
        run_max_bests[3][i] = numpy.amax(run_best_lists[3][i])
    run_min_bests[0][i] = numpy.amin(run_best_lists[0][i])
    run_min_bests[1][i] = numpy.amin(run_best_lists[1][i])
    run_min_bests[2][i] = numpy.amin(run_best_lists[2][i])
    if (four_obj):
        run_min_bests[3][i] = numpy.amin(run_best_lists[3][i])
    run_std_bests[0][i] = numpy.std(run_best_lists[0][i])
    run_std_bests[1][i] = numpy.std(run_best_lists[1][i])
    run_std_bests[2][i] = numpy.std(run_best_lists[2][i])
    if (four_obj):
        run_std_bests[3][i] = numpy.std(run_best_lists[3][i])


fig = plt.figure()

# Fudge factor: plot the averages 10 units offset so we can see them better
plt.errorbar(run_evals + 10, run_average_averages[0],
             [run_average_averages[0] - run_min_averages[0],
              run_max_averages[0] - run_average_averages[0]],
             lw = 0.3, fmt = 'r', label = 'avg cells lit')
plt.errorbar(run_evals, run_average_averages[0], run_std_averages[0],
             lw = 0.6, fmt = 'r')

plt.errorbar(run_evals, run_average_bests[0],
             [run_average_bests[0] - run_min_bests[0],
              run_max_bests[0] - run_average_bests[0]],
             lw = 0.3, fmt = 'b', label = 'best cells lit')
plt.errorbar(run_evals, run_average_bests[0], run_std_bests[0],
             lw = 0.6, fmt = 'b')

plt.errorbar(run_evals + 10, run_average_averages[1],
             [run_average_averages[1] - run_min_averages[1],
              run_max_averages[1] - run_average_averages[1]],
             lw = 0.3, fmt = 'g', label = 'avg bulb vio')
plt.errorbar(run_evals, run_average_averages[1], run_std_averages[1],
             lw = 0.6, fmt = 'g')

plt.errorbar(run_evals, run_average_bests[1],
             [run_average_bests[1] - run_min_bests[1],
              run_max_bests[1] - run_average_bests[1]],
             lw = 0.3, fmt = 'c', label = 'best bulb vio')
plt.errorbar(run_evals, run_average_bests[1], run_std_bests[1],
             lw = 0.6, fmt = 'c')

plt.errorbar(run_evals + 10, run_average_averages[2],
             [run_average_averages[2] - run_min_averages[2],
              run_max_averages[2] - run_average_averages[2]],
             lw = 0.3, fmt = 'm', label = 'avg blk bio')
plt.errorbar(run_evals, run_average_averages[2], run_std_averages[2],
             lw = 0.6, fmt = 'm')

plt.errorbar(run_evals, run_average_bests[2],
             [run_average_bests[2] - run_min_bests[2],
              run_max_bests[2] - run_average_bests[2]],
             lw = 0.3, fmt = 'k', label = 'best blk vio')
plt.errorbar(run_evals, run_average_bests[2], run_std_bests[2],
             lw = 0.6, fmt = 'k')

if (four_obj):
    plt.errorbar(run_evals + 10, run_average_averages[3],
                 [run_average_averages[3] - run_min_averages[3],
                  run_max_averages[3] - run_average_averages[3]],
                 lw = 0.3, fmt = 'y', label = 'avg num bulb')
    plt.errorbar(run_evals, run_average_averages[3], run_std_averages[3],
                 lw = 0.6, fmt = 'y')

    plt.errorbar(run_evals, run_average_bests[3],
                 [run_average_bests[3] - run_min_bests[3],
                  run_max_bests[3] - run_average_bests[3]],
                 lw = 0.3, fmt = 'y', label = 'best num bulb')
    plt.errorbar(run_evals, run_average_bests[3], run_std_bests[3],
                 lw = 0.6, fmt = 'y')


plt.title(fileroot)
plt.xlabel('Evaluations')
plt.ylabel('Fitness')
plt.legend(loc='lower right')

plt.show()

fig.savefig('plots/' + fileroot + '.png', dpi = 600)

# Ref:
# https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation


