import json
with open("colab_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "raise FileNotFoundError(\"No zip files found" in source:
            source = source.replace(
                'raise FileNotFoundError("No zip files found under MyDrive/datasets.")',
                'print("No zip files found — assuming datasets are already extracted and structured.")\n    print("Skipping Cell 8. Proceed to Cell 4 to verify dataset paths.")'
            )
            
            # update source
            cell["source"] = [line if line.endswith('\n') else line + '\n' for line in source.splitlines()]
            if source and not source.endswith('\n'):
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1].rstrip('\n')
            else:
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1].rstrip('\n')

with open("colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
