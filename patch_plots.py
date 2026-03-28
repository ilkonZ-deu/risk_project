import json
import re

with open('project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

plot_count = 1
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'plt.show()' in line:
                # If there's no savefig recently, inject it
                if not any('savefig' in s for s in cell['source']):
                    # Get indentation
                    indent_match = re.match(r'^(\s*)', line)
                    indent = indent_match.group(1) if indent_match else ''
                    new_source.append(f"{indent}import os\n")
                    new_source.append(f"{indent}os.makedirs('plots', exist_ok=True)\n")
                    new_source.append(f"{indent}plt.savefig(f'plots/plot_{plot_count}.png', bbox_inches='tight')\n")
                    plot_count += 1
            new_source.append(line)
        cell['source'] = new_source

with open('project.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
