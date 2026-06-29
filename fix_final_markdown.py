import re

with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    content = f.read()

# 1. Fix the HTML div styling. Convert to Markdown.
old_html_pattern = r'<div align="center".*?</div>'
new_md = """---
> ## 🔍 6. Ablation Study (FeDoRA Components)
> **이 섹션은 메인 실험 결과와 분리된 독립적인 구조 분석(Ablation Study) 섹션입니다.**
> 제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다."""
content = re.sub(old_html_pattern, new_md, content, flags=re.DOTALL)

# 2. Remove the Vision Tables from the Ablation Study section.
ablation_idx = content.find("6. Ablation Study")
if ablation_idx != -1:
    vision_20_idx = content.find("### 20 Clients Setting", ablation_idx)
    if vision_20_idx != -1:
        content = content[:vision_20_idx].strip()

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.write(content)

print("Markdown fixed!")
