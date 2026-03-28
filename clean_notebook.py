import json

with open('project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            # Check if line starts with something like "# 1.", "## 14.", "# 15."
            import re
            if re.match(r'^#+\s*\d+\.\s*', line) or re.match(r'^#\s*Задачи', line) or re.match(r'^#\s*Задание', line):
                continue
            new_source.append(line)
        cell['source'] = new_source

with open('project.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
