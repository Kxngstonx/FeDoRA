import re

with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    content = f.read()

# Replace the emoji and bold asterisks
content = content.replace('**🌟 Ours (FeDoRA)**', 'Ours (FeDoRA)')
# In case there's spaces left weirdly
content = content.replace('🌟 Ours (FeDoRA)', 'Ours (FeDoRA)')

lines = content.split('\n')

new_lines = []
table_lines = []
in_table = False

def format_table(t_lines):
    rows = []
    for line in t_lines:
        parts = line.strip().split('|')
        if len(parts) > 2:
            # handle cases where there might be escaped pipes, but here we assume simple tables
            rows.append([p.strip() for p in parts[1:-1]])
    
    if not rows: return t_lines
    num_cols = len(rows[0])
    max_widths = [0] * num_cols
    for row in rows:
        for i, col in enumerate(row):
            if i < num_cols:
                if set(col.replace(':', '')) == {'-'}:
                    # min width of 3 for dashes
                    max_widths[i] = max(max_widths[i], 3)
                else:
                    max_widths[i] = max(max_widths[i], len(col))
                    
    formatted = []
    for r_idx, row in enumerate(rows):
        formatted_row = "|"
        for i, col in enumerate(row):
            if i >= num_cols: break
            width = max_widths[i]
            if r_idx == 1 and set(col.replace(':', '')) == {'-'}:
                formatted_row += "-" * (width + 2) + "|"
            else:
                if i == 0:
                    formatted_row += f" {col.ljust(width)} |"
                else:
                    formatted_row += f" {col.center(width)} |"
        formatted.append(formatted_row)
    return formatted

for line in lines:
    if line.strip().startswith('|') and line.strip().endswith('|'):
        in_table = True
        table_lines.append(line)
    else:
        if in_table:
            # Format and append
            new_lines.extend(format_table(table_lines))
            table_lines = []
            in_table = False
        new_lines.append(line)

if in_table:
    new_lines.extend(format_table(table_lines))

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.write('\n'.join(new_lines))

