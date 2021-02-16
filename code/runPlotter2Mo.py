# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import scipy.stats as stats
import traceback


"""
Load/calculate/compare/plot the distance from optimal for two experiments.
"""


def read_file(filename):
    """
    Read the results of the log file and return a list of the best of each run.
    """

    curr_run = -1
    lineno = 0
    last_best = -1

    bests = []

    with open(filename, 'r') as reader:
       curr_line = reader.readline()
       while (curr_line):
            lineno += 1
            curr_line = curr_line.strip()

            if (curr_line.startswith('Run')):
                curr_run += 1
                last_best = None
            elif (curr_run > -1):
                if (len(curr_line) == 0):
                    bests.append(last_best)
                else:
                    try:
                        data_list = list(curr_line.split("\t"))
                        last_best = [float(data_list[2]),
                                     float(data_list[4]),
                                     float(data_list[6])]
                    except:
                        print('Problem in line ' + str(lineno) + ': |' + curr_line + '|')
                        traceback.print_exc()
                        pass

            curr_line = reader.readline()

    bests.append(last_best)

    return bests


# Compare two log files

# Yes I hard-coded the optimal solution values.
optimal = []
#optimal = [127, 0, 0]  #d1
optimal = [161, 0, 0]  #d2

#optimal = [127, 0, 0, 31]  #d1
#optimal = [161, 0, 0, 34]  #d2

fileroot1 = 'd2-100-50-fps-plus-trunc'
fileroot2 = 'd2-100-50-fps-plus-trunc-4obj'
combo_fileroot = fileroot1 + '-' + fileroot2

filename1 = 'logs/' + fileroot1 + '.txt'
filename2 = 'logs/' + fileroot2 + '.txt'

# Load bests from files and calculate distances.
bests1 = read_file(filename1)
bests2 = read_file(filename2)
dists1 = []
for best in bests1:
    dists1.append(numpy.linalg.norm(numpy.array(best) - numpy.array(optimal)))
dists2 = []
for best in bests2:
    dists2.append(numpy.linalg.norm(numpy.array(best) - numpy.array(optimal)))


# Calculate F-Test Two-Sample for Variances
mean1 = numpy.mean(dists1)
mean2 = numpy.mean(dists2)
var1 = numpy.var(dists1, ddof=1)
var2 = numpy.var(dists2, ddof=1)
obs = len(dists1)
ft_df = obs - 1
f = var1/var2
ft_p = stats.f.cdf(f, ft_df, ft_df)
alpha = 0.05
fcrit = stats.f.ppf(alpha, ft_df, ft_df)

have_equal_variances = False

print('-----------------------------')
print('\\begin{figure}[H]')
print('\\caption{' + fileroot1 + ' vs. ' + fileroot2 + ' -- Distance from best values to optimal over ' + str(obs) + ' runs}')
print('\\centering')
print('\\includegraphics[width=8cm]{' + combo_fileroot + '.png}')
print('\\label{fig:' + combo_fileroot + '}')
print('\\end{figure}')
print()
print('\\begin{table}[H]')
print('\\centering')
print('\\caption{F-Test for ' + fileroot1 + ' vs. ' + fileroot2 + ' with $\\alpha = ' + str(alpha) + '$}')
print('\\label{tab:ftest-' + combo_fileroot + '}')
print('\\begin{tabular}{lll}')
print('\\hline')
print(' & ' + fileroot1 + ' & ' + fileroot2 + ' \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Mean}     & \\multicolumn{1}{l|}{' + str(mean1) + '} & \\multicolumn{1}{l|}{' + str(mean2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Variance}     & \\multicolumn{1}{l|}{' + str(var1) + '} & \\multicolumn{1}{l|}{' + str(var2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Observations}     & \\multicolumn{1}{l|}{' + str(obs) + '} & \\multicolumn{1}{l|}{' + str(obs) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{df}     & \\multicolumn{1}{l|}{' + str(ft_df) + '} & \\multicolumn{1}{l|}{' + str(ft_df) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{F}     & \\multicolumn{1}{l|}{' + str(f) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{P(F$\leq$f) one-tail}     & \\multicolumn{1}{l|}{' + str(ft_p) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{F Critical one-tail}     & \\multicolumn{1}{l|}{' + str(fcrit) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\end{tabular}')
print('\\end{table}')
print()

if (abs(mean1) > abs(mean2)) and (f < fcrit):
    print('\\noindent abs(mean 1) $>$ abs(mean 2) and F $<$ F Critical implies equal variances.')
    have_equal_variances = True
if (abs(mean1) > abs(mean2)) and (f > fcrit):
    print('\\noindent abs(mean 1) $>$ abs(mean 2) and F $>$ F Critical implies unequal variances.')
    have_equal_variances = False
if (abs(mean1) < abs(mean2)) and (f > fcrit):
    print('\\noindent abs(mean 1) $<$ abs(mean 2) and F $>$ F Critical implies equal variances.')
    have_equal_variances = True
if (abs(mean1) < abs(mean2)) and (f < fcrit):
    print('\\noindent abs(mean 1) $<$ abs(mean 2) and F $<$ F Critical implies unequal variances.')
    have_equal_variances = False
print()

# Calculate T-Test Two-Sample for equal or unequal variances
tt_df = (obs * 2) - 2
tcrit_two_tail = stats.t.ppf(1.0 - (alpha/2), tt_df)
(tstat, tt_p_two_tail) = stats.ttest_ind(dists1, dists2, equal_var=have_equal_variances)

print('\\begin{table}[H]')
print('\\centering')
print('\\caption{t-Test for ' + fileroot1 + ' vs. ' + fileroot2 + ' with ')
if (have_equal_variances):
    print('Equal Variances}')
else:
    print('Unequal Variances}')
print('\\label{tab:ttest-' + combo_fileroot + '}')
print('\\begin{tabular}{lll}')
print('\\hline')
print(' & ' + fileroot1 + ' & ' + fileroot2 + ' \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Mean}     & \\multicolumn{1}{l|}{' + str(mean1) + '} & \\multicolumn{1}{l|}{' + str(mean2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Variance}     & \\multicolumn{1}{l|}{' + str(var1) + '} & \\multicolumn{1}{l|}{' + str(var2) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{Observations}     & \\multicolumn{1}{l|}{' + str(obs) + '} & \\multicolumn{1}{l|}{' + str(obs) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{df}     & \\multicolumn{1}{l|}{' + str(tt_df) + '} & \\multicolumn{1}{l|}{' + str(ft_df) + '} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{t Stat}     & \\multicolumn{1}{l|}{' + str(tstat) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{P(T$\leq$t) two-tail}     & \\multicolumn{1}{l|}{' + str(tt_p_two_tail) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\multicolumn{1}{|l|}{t Critical two-tail}     & \\multicolumn{1}{l|}{' + str(tcrit_two_tail) + '} & \\multicolumn{1}{l|}{} \\\\ \\hline')
print('\\end{tabular}')
print('\\end{table}')
print()

if (abs(tstat) > abs(tcrit_two_tail)):
    print('\\noindent abs(t Stat) $>$ abs(t Critical two-tail) so we reject the null hypothesis -- the two samples are statistically different.')
    print('The average improvement of ' + fileroot1 + ' over ' + fileroot2 + ' is ' + str(mean2 - mean1) + ' closer.')
else:
    print('\\noindent abs(t Stat) $<$ abs(t Critical two-tail) so we accept the null hypothesis -- the two samples are NOT statistically different.')
print('-----------------------------')

# Plot the data
overall_max = numpy.max([numpy.max(dists1), numpy.max(dists2)])
bins = numpy.arange(0, overall_max + 1, (overall_max / 10.0))
plt.hist([dists1, dists2], bins, label = [fileroot1, fileroot2])

plt.title('Dists: ' + fileroot1 + ' and ' + fileroot2)
plt.xlabel('Distance from optimal')
plt.ylabel('Number of average bests')
plt.legend(loc='upper right')

plt.savefig('plots/' + combo_fileroot + '.png', dpi = 600)


