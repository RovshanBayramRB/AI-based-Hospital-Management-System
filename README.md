# AI-based Hospital Management System

An end-to-end decision support pipeline for hospital capacity planning during COVID-19, combining **linear programming**, **deep learning**, and **gradient boosting** into a single chain: work out how many patients a hospital can physically accept, staff it, triage the admitted patients, and assign them to nurses.

The design idea is that these are usually treated as five separate problems, and here the output of each stage becomes the input to the next.

```
1. Resource optimization  ──►  patient capacity
2. Nurse scheduling       ──►  staffing per day
                               │
3a. Chest X-ray CNN  ──► finding ──┐
                                   ├──► 3b. Criticality classifier ──► low / medium / high
   Patient record data ────────────┘
                               │
4. Weighted distribution  ──►  patients assigned to nurses
5. Web interface
```

> **Datasets are confidential and not included.** The public Mexican government COVID-19 dataset and the X-ray image set are referenced by the notebooks but not committed.

---

## Stage 1 — Resource optimization

**`1. Code for Resource Management.ipynb`** · PuLP linear program

Maximizes admissible COVID patients subject to physical resource limits. Decision variables: beds, rooms, patient monitors, oxygen masks, stretchers, and patients.

Constraints encode real operational ratios:

| Constraint | Meaning |
|---|---|
| `x₁ = 4·x₂` | Four beds per room |
| `x₆ ≤ x₁` | One bed per patient |
| `x₆ ≤ 2·x₃` | One monitor serves 2 patients |
| `x₆ ≤ 2.1·x₄` | One oxygen mask serves 2.1 patients |
| `x₆ ≤ 2.4·x₅` | One stretcher serves 2.4 patients |

Each resource is also capped by user-supplied inventory.

**Worked example** (40 beds, 20 rooms, 18 monitors, 20 masks, 10 stretchers): the solver returns `Optimal` with beds 40, rooms 10, monitors 18, masks 20, stretchers 10. **Stretchers are the binding constraint** — at 2.4 patients each, 10 stretchers cap admissions at 24. Buying more beds would change nothing; buying stretchers is the only thing that raises capacity. That's exactly the kind of answer an LP is worth writing for.

> See *Known issues* — the printed capacity figure is twice the model's actual value.

## Stage 2 — Nurse scheduling

**`2. Code for Nurse Scheduling.ipynb`** · PuLP integer program

A classic days-off scheduling problem. Seven integer variables, one per shift pattern — each pattern is five consecutive working days followed by two off — minimizing total nurses subject to per-day minimum coverage.

Framing it by *pattern* rather than by nurse-day keeps the model to seven variables while guaranteeing every nurse gets two consecutive days off, a constraint that would otherwise need explicit encoding.

**Worked example** (minimum coverage Mon–Sun: 22, 21, 23, 22, 20, 19, 17): **29 nurses**, with 7 assigned to the Sat–Wed pattern, 2 to Sun–Thu, and so on.

## Stage 3a — Chest X-ray classification

**`3. Code for X-Ray Classification for Pneumonia.ipynb`** · Keras CNN

| | |
|---|---|
| Classes | COVID19 / NORMAL |
| Train | 1,811 images (545 COVID, 1,266 normal) |
| Test | 484 images (167 COVID, 317 normal) |
| Input | 150×150 RGB |

```
Conv2D(32, 5×5) → MaxPool → Dropout(0.5)
Conv2D(64, 5×5) → MaxPool → Dropout(0.5)
Flatten → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)
```

Adam at `lr=0.001`, up to 20 epochs, with checkpointing on validation loss and early stopping (patience 5). Augmentation is zoom and horizontal flip on the training generator only.

**Test accuracy: 94.83%** (loss 0.143).

## Stage 3b — Criticality classification

**`4. Code for Patient Classification for Criticality Level.ipynb`** · Gradient boosting

Trained on the Mexican government's public COVID-19 record dataset — 566,602 raw records, reduced to 558,779 after filtering out "unknown" sentinel codes.

Criticality is derived from the `icu` field: ICU admission → **high**, ICU declined → **medium**, ICU not applicable → **low**. Fourteen input features cover age, sex, and eleven comorbidities, plus the pneumonia finding that stage 3a supplies.

The raw class distribution is extremely skewed — 440,314 low, 109,161 medium, 9,305 high — so the notebook subsamples to roughly 10,000 per class, giving a balanced 29,849-row training set.

`GradientBoostingClassifier(learning_rate=0.1, max_depth=25, n_estimators=400)`

| | Accuracy |
|---|---|
| Train | 0.792 |
| **Test** | **0.669** |

## Stage 4 — Patient distribution

**`5. Code for Patient Distribution among Nurses.ipynb`**

Loads the pickled classifier, predicts criticality for each incoming patient, converts it to a numeric weight, and assigns patients to nurses round-robin subject to a total-load cap per nurse. Nurses are named `A`, `B`, `C`… with a second letter appended past 26.

A pre-check rejects the run if patients exceed 3.5× the nurse count.

## Stage 5 — Web interface

`app.py` plus a set of Bootstrap templates (`index`, `about`, `faqs`, `prevention`, `contact`, `upload`, `results_chest`, `results_ct`).

---

## Running it
```bash
git clone https://github.com/RovshanBayramRB/AI-based-Hospital-Management-System.git
cd AI-based-Hospital-Management-System
pip install pulp pandas numpy scikit-learn seaborn matplotlib tensorflow
jupyter notebook
```

---

## Repository structure

```
.
├── Python Codes/
│   ├── 1. Code for Resource Management.ipynb
│   ├── 2. Code for Nurse Scheduling.ipynb
│   ├── 3. Code for X-Ray Classification for Pneumonia.ipynb
│   ├── 4. Code for Patient Classification for Criticality Level.ipynb
│   └── 5. Code for Patient Distribution among Nurses.ipynb
│
├── app.py                  # Flask app (see Known issues)
├── requirements.txt
│
├── index.html              # Templates
├── about.html
├── faqs.html
├── prevention.html
├── contact.html
├── upload.html
├── results_chest.html
├── results_ct.html
│
├── css/  js/  scss/  fonts/   # Front-end assets
├── inc/sendemail.php
└── README.md
```
