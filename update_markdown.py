with open('final_performance_table_ema03_with_superglue.md', 'r') as f:
    content = f.read()

# Update running acc
content = content.replace('*0.6619*', '*0.7342*')
content = content.replace('*0.6571*', '*0.7155*')

# Make Ablation Study visually distinct
old_header = "## 📚 6. Ablation Study (FeDoRA Components)"
new_header = """<br>

<div align="center" style="background-color: #f6f8fa; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de;">
  <h2 style="color: #0969da; margin-top: 0;"> 🔍 6. Ablation Study (FeDoRA Components) </h2>
  <span style="color: #57606a; font-size: 0.95em;">이 섹션은 메인 실험 결과와 <b>분리된 독립적인 구조 분석(Ablation Study)</b> 섹션입니다.<br>제안 기법(FeDoRA) 내부의 각 요소(FlexLoRA, FFALoRA)가 성능에 미치는 영향을 개별적으로 비교합니다.</span>
</div>

<br>"""

if old_header in content:
    content = content.replace(old_header, new_header)

with open('final_performance_table_ema03_with_superglue.md', 'w') as f:
    f.write(content)
