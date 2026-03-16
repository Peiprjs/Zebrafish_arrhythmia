with open('python/data_analysis.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith('from scipy.stats import f_oneway, linregress, norm, t as student_t, ttest_rel'):
        new_lines.append('from scipy.stats import f_oneway, linregress, norm, t as student_t, ttest_rel\nfrom functions import paired_ttest_pvalue\n')
    elif line.startswith('def paired_ttest_pvalue(values, reference_values, paired=True):'):
        skip = True
    elif skip and line.strip() == 'def add_control_pvalues(full_df):':
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open('python/data_analysis.py', 'w') as f:
    f.writelines(new_lines)
