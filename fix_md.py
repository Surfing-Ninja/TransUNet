import json
with open("colab_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        for i, line in enumerate(cell["source"]):
            if "To switch datasets:" in line or "DATASET_NAME" in line:
                if "**To switch datasets:** Go back to **Cell 8**" in line:
                    cell["source"][i] = "> To train a single dataset: Go to Cell 18, set `TRAIN_ONLY = \"tcga_lgg\"` (or whichever dataset you want), then re-run Cell 18.\n"

with open("colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
