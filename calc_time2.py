import re
import glob
import os
from datetime import datetime

log_files = glob.glob("logs/vit_cifar100_svhn_experiments/**/*.log", recursive=True)
completed_logs = []
for f in log_files:
    with open(f, 'r') as file:
        content = file.read()
        if "ROUND 49/50" in content or "ROUND 50/50" in content:
            completed_logs.append(f)

if not completed_logs:
    print("No completed logs found to calculate time.")
else:
    # Get the start and end times from the first completed log
    with open(completed_logs[0], 'r') as file:
        lines = file.readlines()
        
    start_time = None
    end_time = None
    
    for line in lines:
        match = re.search(r'\[\d{2}-\d{2} (\d{2}:\d{2}:\d{2})\]', line)
        if match:
            t = datetime.strptime(match.group(1), "%H:%M:%S")
            if start_time is None:
                start_time = t
            end_time = t
            
    if start_time and end_time:
        duration_sec = (end_time - start_time).total_seconds()
        if duration_sec < 0: # crossed midnight
            duration_sec += 24 * 3600
        
        print(f"Time per experiment (50 rounds): {duration_sec/60:.1f} minutes")
        
        # 48 total experiments, 2 running concurrently -> 24 batches
        # We are currently in the 2nd batch
        # So 22 batches remaining + remaining time of current batch
        # Let's say 23 batches remaining in total for simplicity
        total_remaining_hours = (23 * duration_sec) / 3600
        print(f"Estimated remaining time: {total_remaining_hours:.1f} hours")
    else:
        print("Could not parse timestamps.")

# Now fix the markdown
import re

with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    content = f.read()

# 1. Remove 20 clients and 50 clients settings tables from Ablation Study section
# 2. Fix the HTML styling to use markdown

# First, fix the HTML block
old_html = """<div align="center" style="background-color: #f6f8fa; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de;">
  <h2 style="color: #0969da; margin-top: 0;"> 🔍 6. Ablation Study (FeDoRA Components) </h2>
  <span style="color: #57606a; font-size: 0.95em;">이 섹션은 메인 실험 결과와 <b>분리된 독립적인 구조 분석(Ablation Study)</b> 섹션입니다.<br>제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다.</span>
</div>"""

new_markdown = """---

> ## 🔍 6. Ablation Study (FeDoRA Components)
> **이 섹션은 메인 실험 결과와 분리된 독립적인 구조 분석(Ablation Study) 섹션입니다.**
> 제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다."""

content = content.replace(old_html, new_markdown)

# Second, remove the Vision tables from Ablation Study
# The ablation study starts at '## 🔍 6. Ablation Study' (or similar)
ablation_index = content.find("6. Ablation Study")
if ablation_index != -1:
    vision_20_index = content.find("### 20 Clients Setting", ablation_index)
    if vision_20_index != -1:
        # We want to remove everything from '### 20 Clients Setting' to the end of the file
        content = content[:vision_20_index].strip()

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.write(content)

