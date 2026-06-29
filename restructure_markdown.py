import re

with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    lines = f.readlines()

new_lines = []
ablation_sections = []

current_table_headers = []
current_ablation_rows = []

in_table = False
table_title = ""

for i, line in enumerate(lines):
    if line.startswith("## 📚"):
        table_title = line.strip()
    elif line.startswith("### 🌱"):
        table_title = line.strip()
    elif line.startswith("### 20 Clients Setting") or line.startswith("### 50 Clients Setting"):
        table_title = line.strip()
        
    if line.startswith("| Method"):
        in_table = True
        current_table_headers = [line, lines[i+1]]
        current_ablation_rows = []
        new_lines.append(line)
        continue
    
    if in_table and line.startswith("|---"):
        new_lines.append(line)
        continue
        
    if in_table and line.startswith("|"):
        if "Ours (FeDoRA)" in line:
            # Fix spacing: remove 1 space before the first | (actually before the pipe after the name)
            # Find the first pipe after the name
            parts = line.split('|')
            if len(parts) > 2:
                col1 = parts[1]
                # Remove one space from the right of col1 if it ends with spaces
                if col1.endswith(' '):
                    col1 = col1[:-1]
                parts[1] = col1
                line = '|'.join(parts)
            new_lines.append(line)
            current_ablation_rows.append(line)
        elif "FL+DoRA" in line:
            current_ablation_rows.append(line)
            # Do not append to new_lines (remove from main table)
        else:
            new_lines.append(line)
    else:
        if in_table: # table ended
            if len(current_ablation_rows) > 0:
                ablation_sections.append((table_title, current_table_headers, current_ablation_rows))
            in_table = False
        new_lines.append(line)

if in_table:
    if len(current_ablation_rows) > 0:
        ablation_sections.append((table_title, current_table_headers, current_ablation_rows))

# Append Ablation Study
new_lines.append("\n## 📚 6. Ablation Study (FeDoRA Components)\n\n")

for title, headers, rows in ablation_sections:
    if len(rows) > 0:
        if title.startswith("###"):
            new_lines.append(f"{title}\n")
        else:
            new_lines.append(f"### {title.replace('## 📚 ', '')}\n")
            
        new_lines.extend(headers)
        new_lines.extend(rows)
        new_lines.append("\n")

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.writelines(new_lines)
