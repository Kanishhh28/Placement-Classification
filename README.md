# Will this student get placed?

The placement cell has a spreadsheet of students who registered for placements. Each row is one student. `placed` is 1 if that student received an offer through the cell, and 0 if not.
che
Build a classifier that predicts `placed` for a new list of registered students.

`data/train.csv` is the training data. We score you on a **separate list** that is not in this folder.

## Columns

| Column | What it is |
| --- | --- |
| `student_id` | Student id |
| `department` | Branch: CSE, IT, ECE, EEE, MECH, or CIVIL |
| `city_tier` | 1, 2, or 3 |
| `cgpa` | Cumulative GPA |
| `backlogs` | Active backlogs |
| `internship_months` | Length of internship, in months |
| `coding_score` | Internal coding test, out of 100 |
| `projects` | Number of projects listed |
| `workshops` | Number of workshops attended |
| `status_code` | A field stored on the placement portal |
| `placed` | 1 if placed, 0 otherwise. Training file only |

## Evaluation

We run `predict.py` on the separate list and compute the **F1 score for `placed = 1`**.

We also read `analysis.ipynb`: what you checked in the data, what you chose to use in the model, and why.

## What to submit

```
analysis.ipynb
predict.py
requirements.txt
```

Train the model before you submit. `predict.py` should load that model and score a new file. It should not fit a new model.

```bash
python predict.py --input some_students.csv --output predictions.csv
```

## Output format

`predictions.csv`:

```
student_id,placed
S00001,0
S00002,1
```

One row per student in the input. `placed` is 0 or 1.

`data/sample_submission.csv` shows this layout. The 0/1 values in that file are placeholders, not correct answers.

The file we score has the same columns except `placed`. Some cells can be empty.
