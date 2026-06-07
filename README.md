# MT Evaluation Metric Based on Perplexity

An automatic Machine Translation evaluation metric that combines [BLASER](https://github.com/facebookresearch/stopes/tree/main/stopes/eval/blaser) scores (scale 1–5) with perplexity to penalize disfluent translations. A perplexity-derived confidence weight modulates BLASER (or whatever MT metric) so that segments with high perplexity (low fluency) receive lower combined scores.

## Formula

The perplexity weight is computed using a specific normalization (min-max, zscore or robust). The **min-max normalization** in log-space (the default `--method minmax`) is as follows:

$$w = \exp\!\left(-\frac{\log \text{PPL} - \log \text{PPL}_{\min}}{\log \text{PPL}_{\max} - \log \text{PPL}_{\min}}\right)$$

This yields $w \to 1$ for low-perplexity (fluent) outputs and $w \to 0$ for high-perplexity (disfluent) outputs. The combined metric is then:

$$\text{BLASER}_{\text{PPL}} = B \cdot w^{\alpha}$$

The final score is rescaled to the **[1, 5]** range by default.

| Symbol | Meaning |
|--------|---------|
| $w$ | Fluency confidence weight derived from perplexity (0, 1] |
| $B$ | BLASER score ∈ [1, 5] |
| $\alpha$ | Controls how much perplexity influences the final score (0 = ignore PPL, 1 = full weight) |
|  $\text{PPL}\_{\min}$, $\text{PPL}\_{\max}$ | Minimum and maximum perplexity values observed in the dataset |

## Repository Contents

| File | Description |  
|------|-------------|
| `metric_ppl.py` | Python script implementing the combined metric |
| `nos.tsv` | BLASER + perplexity scores for **TradutorNós** (334 segments) |
| `ft.tsv` | BLASER + perplexity scores for **Carvalho_Finetuning** (334 segments) |
| `sal.tsv` | BLASER + perplexity scores for **SalamandraTA** (334 segments) |
| `LICENSE` | GNU General Public License v3.0 |

Each TSV file contains two tab-separated columns (with a header row):

| Column 1 | Column 2 |
|----------|----------|
| BLASER score (1–5) | Perplexity |

To generate the perplexity values, you can use the script `ppl.py` by taking as an argument the sentences of the test dataset.

## Requirements

- Python 3.10+
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)

Install dependencies:

```bash
pip install numpy pandas scipy
```

## Usage

```bash
python metric_ppl.py <input.tsv> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | `0.5` | Perplexity influence weight (0–1) |
| `--method` | `zscore` | PPL normalisation method: `zscore`, `minmax`, or `robust` |
| `--no-rescale` | off | Do not rescale the combined score to [1, 5] |
| `--output` | — | Path to save full results as TSV |
| `--sensitivity` | off | Print alpha sensitivity analysis |

### Examples

```bash
# Evaluate TradutorNós with a mild perplexity penalty
python metric_ppl.py nos.tsv --alpha 0.15 --output nos15.tsv

# Evaluate SalamandraTA with default settings
python metric_ppl.py sal.tsv

# Compare alpha sensitivity for Carvalho_Finetuning
python metric_ppl.py ft.tsv --alpha 0.3 --sensitivity

# Use robust normalisation (less sensitive to outliers)
python metric_ppl.py nos.tsv --alpha 0.5 --method robust --output nos_robust.tsv
```

### Output

The script prints to stdout:

1. **Descriptive statistics** for BLASER, perplexity, PPL weight, and combined scores
2. **Correlation report** (Pearson and Spearman) between all metric pairs
3. **Configuration summary**
4. **Alpha sensitivity analysis** (with `--sensitivity`)
5. **Sample preview** (first 10 rows)
6. **Final average score** with 95% confidence interval and qualitative band (Poor / Fair / Good / Very good / Excellent)

When `--output` is provided, the full table (including `ppl_weight`, `combined_raw`, and `combined_final` columns) is saved as a TSV file.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
