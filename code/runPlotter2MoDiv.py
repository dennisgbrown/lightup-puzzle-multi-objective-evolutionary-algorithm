# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import scipy.stats as stats


"""
Load/calculate/compare/plot the diversity for two experiments.
"""


def read_file(filename):
    """
    Read final best front from file.
    """

    members = []

    with open(filename, 'r') as reader:
        curr_line = reader.readline()
        while (curr_line):
            curr_line = curr_line.strip()
            data_list = list(curr_line.split("\t"))
            member = [float(data_list[0]), float(data_list[1]),
                      float(data_list[2]), float(data_list[3])]
            members.append(member)
            curr_line = reader.readline()

    return members


# Compare two log files

fileroot1 = 'd2-100-50-fps-plus-trunc'
fileroot2 = 'd2-100-50-fps-plus-trunc-4obj'
combo_fileroot = fileroot1 + '-' + fileroot2

filename1 = 'data/' + fileroot1 + '.txt'
filename2 = 'data/' + fileroot2 + '.txt'

# Load front members from file
mems1 = read_file(filename1)
mems2 = read_file(filename2)

# Calculate distances with the crowding algorithm
dists1 = numpy.zeros(len(mems1) - 2)
for curr_obj in range(0, 3):
    sorted_mems = sorted(mems1, key = lambda i: i[curr_obj])
    min_obj = sorted_mems[0][curr_obj]
    max_obj = sorted_mems[-1][curr_obj]
    for i in range(1, len(sorted_mems) - 1):
        if ((max_obj - min_obj) != 0):
            prev_obj = sorted_mems[i - 1][curr_obj]
            next_obj = sorted_mems[i + 1][curr_obj]
            dists1[i - 1] += (next_obj - prev_obj) / (max_obj - min_obj)

dists2 = numpy.zeros(len(mems2) - 2)
for curr_obj in range(0, 3):
    sorted_mems = sorted(mems2, key = lambda i: i[curr_obj])
    min_obj = sorted_mems[0][curr_obj]
    max_obj = sorted_mems[-1][curr_obj]
    for i in range(1, len(sorted_mems) - 1):
        if ((max_obj - min_obj) != 0):
            prev_obj = sorted_mems[i - 1][curr_obj]
            next_obj = sorted_mems[i + 1][curr_obj]
            dists2[i - 1] += (next_obj - prev_obj) / (max_obj - min_obj)

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
print('\\caption{' + fileroot1 + ' vs. ' + fileroot2 + ' -- Diversity (Crowding distance)}')
print('\\centering')
print('\\includegraphics[width=8cm]{' + combo_fileroot + '-diversity.png}')
print('\\label{fig:' + combo_fileroot + '-diversity}')
print('\\end{figure}')
print()
print('\\begin{table}[H]')
print('\\centering')
print('\\caption{F-Test for ' + fileroot1 + ' vs. ' + fileroot2 + ' with $\\alpha = ' + str(alpha) + '$}')
print('\\label{tab:ftest-' + combo_fileroot + '-diversity}')
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
print('\\label{tab:ttest-' + combo_fileroot + '-diversity}')
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
    print('The average improvement of ' + fileroot1 + ' over ' + fileroot2 + ' is ' + str(mean1 - mean2) + ' more diverse.')
else:
    print('\\noindent abs(t Stat) $<$ abs(t Critical two-tail) so we accept the null hypothesis -- the two samples are NOT statistically different.')
print('-----------------------------')

# Plot the data
overall_max = numpy.max([numpy.max(dists1), numpy.max(dists2)])
bins = numpy.arange(0, overall_max + 1, (overall_max / 10.0))
plt.hist([dists1, dists2], bins, label = [fileroot1, fileroot2])

plt.title('Dists: ' + fileroot1 + ' and ' + fileroot2)
plt.xlabel('Diversity (Crowding Distance)')
plt.ylabel('Number of average bests')
plt.legend(loc='upper right')

plt.savefig('plots/' + combo_fileroot + '-diversity.png', dpi = 600)


