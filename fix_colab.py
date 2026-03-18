import json

with open("colab_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Remove importlib.reload from cells 12-22. 
        # (Actually, users specified "from cells 12 onwards", so anything after cell 10).
        # We also need to remove the lines "import importlib\n" and "import config\n" possibly, 
        # but keep "from config import CFG" or just leave it. The user explicitly showed removing them.
        if idx >= 11:  # Cell 12 is index 11
            source = source.replace("import importlib\n", "")
            source = source.replace("import config\n", "")
            source = source.replace("importlib.reload(config)\n", "")
        
        # 2. Fix test ratio in cell 8 Auto-unzip
        source = source.replace("test_ratio=0.2", "test_ratio=0.1")
        
        # 3. Fix keyword ct
        source = source.replace('"covid_ct": ["covid", "ct", "lung", "infection"],', '"covid_ct": ["covid", "lung", "infection"],')
        
        cell["source"] = [line if line.endswith('\n') else line + '\n' for line in source.splitlines()]
        if source and not source.endswith('\n'):
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip('\n')
        else:
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip('\n')
                
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "To switch datasets: Go back to Cell 8, change `DATASET_NAME`" in source:
            source = source.replace(
                "To switch datasets: Go back to Cell 8, change `DATASET_NAME`",
                "To train a single dataset: Go to Cell 18, set `TRAIN_ONLY = \"tcga_lgg\"` (or whichever dataset you want), then re-run Cell 18."
            )
        # update source
        cell["source"] = [line + '\n' for line in source.split('\n')]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip('\n')

with open("colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
