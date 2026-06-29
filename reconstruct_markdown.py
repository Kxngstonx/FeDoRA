with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    lines = f.readlines()

new_lines = []
ablation_sections = []

current_table_headers = []
current_ablation_rows = []

in_table = False
table_title = ""

# Delete anything from Vision onwards
filtered_lines = []
for line in lines:
    if "## 📚 5. Vision Benchmarks" in line:
        break
    filtered_lines.append(line)

for i, line in enumerate(filtered_lines):
    if line.startswith("## 📚"):
        table_title = line.strip()
    elif line.startswith("### 🌱"):
        table_title = line.strip()
        
    if line.startswith("| Method"):
        in_table = True
        current_table_headers = [line, filtered_lines[i+1]]
        current_ablation_rows = []
        new_lines.append(line)
        continue
    
    if in_table and line.startswith("|---"):
        new_lines.append(line)
        continue
        
    if in_table and line.startswith("|"):
        line = line.replace('**🌟 Ours (FeDoRA)**', 'Ours (FeDoRA)').replace('🌟 Ours (FeDoRA)', 'Ours (FeDoRA)')
        if "Ours (FeDoRA)" in line:
            new_lines.append(line)
            current_ablation_rows.append(line)
        elif "FL+DoRA" in line:
            current_ablation_rows.append(line)
            # Do not append to main table
        else:
            new_lines.append(line)
    else:
        if in_table:
            if len(current_ablation_rows) > 0:
                ablation_sections.append((table_title, current_table_headers, current_ablation_rows))
            in_table = False
        new_lines.append(line)

if in_table:
    if len(current_ablation_rows) > 0:
        ablation_sections.append((table_title, current_table_headers, current_ablation_rows))

# Append Vision Tables
vision_section = """
## 📚 5. Vision Benchmarks (CIFAR-100 & SVHN)
*Note: Results are based on the currently executed experiments (Lower Budget, Seed 42). Experiments marked with an asterisk (*) are still in progress.*

### 20 Clients Setting
| Method        | Rank | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID) | SVHN (Non-IID) |
|---------------|------|-----------------|---------------------|------------|----------------|
| Ours (FeDoRA) |  32  |        -        |          -          |     -      |       -        |
| FedEx-LoRA    |  32  |      0.9249     |          -          |     -      |       -        |
| FedIT         |  32  |      0.9059     |          -          |     -      |       -        |
| FlexLoRA      |  32  |     *0.7342*    |          -          |     -      |       -        |
| FFA-LoRA      |  64  |     *0.7155*    |          -          |     -      |       -        |
| RAVAN         | 110  |        -        |          -          |     -      |       -        |

### 50 Clients Setting
| Method        | Rank | CIFAR-100 (IID) | CIFAR-100 (Non-IID) | SVHN (IID) | SVHN (Non-IID) |
|---------------|------|-----------------|---------------------|------------|----------------|
| Ours (FeDoRA) |  32  |        -        |          -          |     -      |       -        |
| FedEx-LoRA    |  32  |        -        |          -          |     -      |       -        |
| FedIT         |  32  |        -        |          -          |     -      |       -        |
| FlexLoRA      |  32  |        -        |          -          |     -      |       -        |
| FFA-LoRA      |  64  |        -        |          -          |     -      |       -        |
| RAVAN         | 110  |        -        |          -          |     -      |       -        |

"""
new_lines.extend(vision_section.split('\n'))
new_lines = [l + '\n' if not l.endswith('\n') else l for l in new_lines]

# Append Ablation Study
new_lines.append("---\n\n")
new_lines.append("> ## 🔍 6. Ablation Study (FeDoRA Components)\n")
new_lines.append("> **이 섹션은 메인 실험 결과와 분리된 독립적인 구조 분석(Ablation Study) 섹션입니다.**\n")
new_lines.append("> 제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다.\n\n")

for title, headers, rows in ablation_sections:
    if len(rows) > 0:
        if title.startswith("###"):
            new_lines.append(f"{title}\n")
        else:
            new_lines.append(f"### {title.replace('## 📚 ', '')}\n")
            
        new_lines.extend(headers)
        new_lines.extend(rows)
        new_lines.append("\n")

# Format tables
content = "".join(new_lines)
final_lines = content.split('\n')
formatted_lines = []
t_lines = []
in_t = False

def format_table(t_l):
    rows = []
    for l in t_l:
        p = l.strip().split('|')
        if len(p) > 2:
            rows.append([x.strip() for x in p[1:-1]])
    if not rows: return t_l
    nc = len(rows[0])
    mw = [0]*nc
    for r in rows:
        for i,c in enumerate(r):
            if i < nc:
                if set(c.replace(':','')) == {'-'}: mw[i] = max(mw[i], 3)
                else: mw[i] = max(mw[i], len(c))
    ret = []
    for ri, r in enumerate(rows):
        fr = "|"
        for i,c in enumerate(r):
            if i >= nc: break
            w = mw[i]
            if ri == 1 and set(c.replace(':','')) == {'-'}:
                fr += "-"*(w+2) + "|"
            else:
                if i == 0: fr += f" {c.ljust(w)} |"
                else: fr += f" {c.center(w)} |"
        ret.append(fr)
    return ret

for l in final_lines:
    if l.strip().startswith('|') and l.strip().endswith('|'):
        in_t = True
        t_lines.append(l)
    else:
        if in_t:
            formatted_lines.extend(format_table(t_lines))
            t_lines = []
            in_t = False
        formatted_lines.append(l)

if in_t:
    formatted_lines.extend(format_table(t_lines))

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.write('\n'.join(formatted_lines))

print("Reconstruction complete.")
