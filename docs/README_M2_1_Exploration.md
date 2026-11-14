# M2.1 EGFR/HER2 data exploration notebook

Copy `M2_1_EGFR_HER2_Data_Exploration.ipynb` into your repository's `notebooks/` folder.

## Start in VS Code / PowerShell

```powershell
New-Item -ItemType Directory -Force .\notebooks | Out-Null
# Copy the downloaded notebook into .\notebooks\
code .\notebooks\M2_1_EGFR_HER2_Data_Exploration.ipynb
```

Choose the `ai-dual-target-drug-discovery` Python/uv environment as the kernel. Install Jupyter and ipykernel in your VS Code environment if prompted; data-analysis dependencies (`pandas`, `numpy`, `matplotlib`, `rdkit`) are declared in the project manifest and/or its dev group.

Run cells top to bottom from the repository root or `notebooks/` directory. This notebook only reads:

- `data/interim/M2_1_scaffold_split_molecules.csv`
- `data/interim/M2_1_scaffold_split_manifest.csv`
- `reports/audits/M2_1_scaffold_split_summary.json`

The notebook checks the recorded manifest SHA-256 `05968ca1ed2a16ca8165e8a28ececa0f361fc9e61a64ac2e69abb1e0e4b7423a`. If it doesn't match, investigate the split; do not silently change the expected value.

It reads the actual private local data at runtime (the notebook contains no private BindingDB-derived records). The test partition appears only in partition/label-mask/scaffold integrity summaries; test potency values and activity classes are not inspected. It does not create any model or write back to the frozen dataset.
