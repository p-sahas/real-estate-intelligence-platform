import json

notebook_path = "notebooks/03_intelligence_layers.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell.get('source', []):
            if "embeddings=embeddings" in line:
                line = line.replace("embeddings=embeddings", "embedding=embeddings")
                changed = True
            new_source.append(line)
        cell['source'] = new_source

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully: replaced 'embeddings' with 'embedding'.")
else:
    print("No changes needed or line not found.")
