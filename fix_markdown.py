with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    lines = f.readlines()

new_lines = []
in_ablation = False

for line in lines:
    if "## 📚 6. Ablation Study" in line:
        in_ablation = True
    
    if not in_ablation and "FL+DoRA" in line:
        # Skip this line (remove from main tables)
        continue
    
    # If it's the "Ours" line, reduce one space before the pipe to fix the emoji width issue
    if "Ours (FeDoRA)" in line:
        parts = line.split('|')
        if len(parts) > 2:
            col1 = parts[1]
            if col1.endswith(' '):
                parts[1] = col1[:-1]
            line = '|'.join(parts)
            
    new_lines.append(line)

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.writelines(new_lines)
