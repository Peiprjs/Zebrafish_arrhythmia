with open('python/gui_app.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip() == 'paired_ttest_pvalue,':
        continue
    if line.strip() == 'from data_analysis import (':
        new_lines.append('from functions import paired_ttest_pvalue\n')
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('python/gui_app.py', 'w') as f:
    f.writelines(new_lines)
