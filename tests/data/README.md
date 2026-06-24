# Test Data

Ground-truth dataset for the Simpute guard test suite (`tests/guard.py`).

## `test.csv`

Synthetic student survey-style tabular data (~50,000 rows). Columns:

| Column | Type | Description |
|--------|------|-------------|
| `Student_ID` | identifier | Row ID (excluded from imputation) |
| `Major_Category` | categorical | Academic major group |
| `Year_of_Study` | categorical | Freshman through Senior |
| `Pre_Semester_GPA` | continuous | GPA before the semester |
| `Weekly_GenAI_Hours` | continuous | Weekly generative-AI usage hours |
| `Primary_Use_Case` | categorical | Main GenAI use case |
| `Prompt_Engineering_Skill` | categorical | Beginner / Intermediate / Advanced |
| `Tool_Diversity` | discrete numerical | Count of tools used |
| `Paid_Subscription` | boolean | Whether the student pays for a subscription |
| `Traditional_Study_Hours` | continuous | Weekly non-AI study hours |

Guard tests mask a fraction of values, run `Simpute`, and compare imputed cells back to the hidden ground truth.

## Using your own data

You can replace `test.csv` with any CSV you like. Keep these in mind:

- Use a header row with column names.
- Mix numerical, categorical, and boolean columns if you want full guard coverage.
- Keep at least one ID-style column and list it in `EXCLUDECOLUMNS` inside `tests/guard.py` (default: `Student_ID`).
- Aim for enough rows that models can train (hundreds or more works best).

After swapping the file, re-run:

```bash
pytest tests/guard.py -v
python tests/guard.py
```

If column types or cardinality change a lot, you may need to relax thresholds in `tests/guard.py` (`NUMERICALTOLERANCE`, accuracy baselines) so expectations match your dataset.

Regenerate README validation plots from the new data:

```bash
python scripts/generate_plots.py
```
