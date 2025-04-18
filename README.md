# 📡 RLFAP Solver - README

## 🧩 Overview
The **Radio Link Frequency Assignment Problem (RLFAP)** is a constraint satisfaction problem where frequencies must be assigned to radio transmitters such that constraints are satisfied. These constraints typically represent interference avoidance and assignment requirements.

This project explores and compares the performance of four algorithms:
- **FC (Forward Checking)**
- **MAC (Maintaining Arc Consistency)**
- **FC-CBJ (Forward Checking with Conflict-directed Backjumping)**
- **Min-Conflicts (local search with a max steps limit)**

---

## 🛠️ Implementation Details

Each algorithm processes CSP instances and outputs:
- Number of constraints checked
- Number of variable assignments
- Execution time
- Whether a solution was found (Satisfiable: YES/NO)
- Number of conflicts (for Min-Conflicts only)
---
## 🤖 Why Use DOM/WDEG Heuristic?

The **DOM/WDEG (Domain over Weighted Degree)** heuristic was used to improve variable ordering during search. It selects the variable with the smallest ratio of:

```
DOM / WDEG = Current Domain Size / Sum of Weights of Related Constraints
```

- **DOM** helps prioritize variables with fewer remaining options.
- **WDEG** focuses on variables involved in more "failing" constraints (weights increase on conflict).

This dynamic approach improves efficiency by:
- Guiding the solver to tackle the most constrained (and problematic) variables first.
- Helping to prune the search space earlier.
- Adapting as conflicts are discovered.

## 📊 Performance Tables

### ➤ FC with DOM/WDEG

| Instance     | Constraints Checked | Assignments | Time (s) | Satisfiable |
|--------------|----------------------|-------------|----------|-------------|
| 2-f24        | 19954               | 229         | 0.024    | YES         |
| 2-f25        | 31549767            | 167451      | 18.935   | NO          |
| 3-f10        | 8802782             | 60301       | 5.930    | YES         |
| 3-f11        | 101187403           | 603812      | 63.023   | NO          |
| 6-w2         | 4126745             | 39610       | 2.873    | NO          |
| 7-w1-f4      | 26523683            | 332616      | 26.160   | YES         |
| 7-w1-f5      | -                   | -           | TIMEOUT  | NO          |
| 8-f10        | -                   | -           | TIMEOUT  | NO          |
| 8-f11        | 20010090            | 175899      | 19.198   | NO          |
| 11           | 188770093           | 1080442     | 146.861  | YES         |
| 14-f27       | 4799797             | 90220       | 7.684    | YES         |
| 14-f28       | 891653              | 17193       | 3.090    | NO          |

---

### ➤ MAC with DOM/WDEG

| Instance     | Constraints Checked | Assignments | Time (s) | Satisfiable |
|--------------|----------------------|-------------|----------|-------------|
| 2-f24        | 374360              | 364         | 0.258    | YES         |
| 2-f25        | 15187720            | 7931        | 11.779   | NO          |
| 3-f10        | 748113              | 406         | 0.428    | YES         |
| 3-f11        | 699220411           | 126582      | 407.152  | NO          |
| 6-w2         | 12333822            | 2344        | 5.976    | NO          |
| 7-w1-f4      | 1039773             | 646         | 0.526    | YES         |
| 7-w1-f5      | 48749962            | 14380       | 23.972   | NO          |
| 8-f10        | 271400349           | 200349      | 216.830  | YES         |
| 8-f11        | 16264990            | 5909        | 10.959   | NO          |
| 11           | 2788039             | 1050        | 1.541    | YES         |
| 14-f27       | 6536878             | 7791        | 6.217    | YES         |
| 14-f28       | 11093318            | 17888       | 13.127   | NO          |

---

### ➤ FC-CBJ with DOM/WDEG

| Instance     | Constraints Checked | Assignments | Time (s) | Satisfiable |
|--------------|----------------------|-------------|----------|-------------|
| 2-f24        | 155972              | 1053        | 0.110    | YES         |
| 2-f25        | 38835782            | 190382      | 21.616   | NO          |
| 3-f10        | 187133              | 1233        | 0.150    | YES         |
| 3-f11        | 28260167            | 139512      | 17.089   | NO          |
| 6-w2         | 4149176             | 36798       | 2.961    | NO          |
| 7-w1-f4      | 138099              | 1625        | 0.203    | YES         |
| 7-w1-f5      | 137356948           | 2371294     | 177.013  | NO          |
| 8-f10        | 100388364           | 818411      | 88.636   | YES         |
| 8-f11        | 5430893             | 37678       | 5.260    | NO          |
| 11           | 44512204            | 224103      | 32.335   | YES         |
| 14-f27       | 802258              | 9990        | 1.991    | YES         |
| 14-f28       | 3578738             | 42237       | 7.623    | NO          |

---

### ➤min-conflicts

| Instance     | Constraints Checked | Assignments | Time (s) | Satisfiable | Conflicts |
|--------------|----------------------|-------------|----------|-------------|-----------|
| 2-f24        | 3048593             | 1831        | 1.4      | NO          | 17        |
| 2-f25        | 3009404             | 1200        | 1.4      | NO          | 10        |
| 3-f10        | 27357093            | 65000       | 3.0      | NO          | 44        |
| 3-f11        | 115335217           | 610011      | 3.0      | NO          | 57        |
| 6-w2         | 9540391             | 42571       | 0.9      | NO          | 85        |
| 7-w1-f4      | 31041051            | 337616      | 0.9      | NO          | 73        |
| 7-w1-f5      | 50301599            | 24915       | 0.9      | NO          | 74        |
| 8-f10        | 44741043            | 18343       | 0.9      | NO          | 215       |
| 8-f11        | 44505043            | 181998      | 4.4      | NO          | 259       |
| 11           | 217577013           | 1086842     | 5.1      | NO          | 10        |
| 14-f27       | 33504558            | 97800       | 5.2      | NO          | 370       |
| 14-f28       | 20019610            | 23279       | 5.2      | NO          | 466       |

---

## 📈 Observations

- **MAC** checks more constraints but performs fewer assignments than FC and FC-CBJ. It's faster in large-domain problems but slower when constraint density increases.
- **FC** is slower than **FC-CBJ**, which is expected since CBJ reduces backtracking.
- All three algorithms (FC, MAC, FC-CBJ) yielded consistent satisfiability results.
- **Min-Conflicts**:
  - Performance is heavily dependent on the max number of steps.
  - It randomly selects variables, often leading to dead ends.
  - Conflicts increase with problem size.
  - Does many more assignments and constraint checks than the others.
  > 💡 Note: All data for Min-Conflicts reflects average metrics over 5 runs with `max_steps=1000`. Despite the high number of steps, the algorithm could not find a solution in any instance within the allowed limits.


---

## 🧪 Metrics Used

- **Constraints Checked** – Number of times a constraint was evaluated.
- **Number of Assignments** – Total variable assignments attempted.
- **Time Taken** – Execution duration in seconds.
- **Satisfiable** – Whether a valid solution was found.
- **Number of Conflicts** – Only tracked in Min-Conflicts.

---

## Code citation
  I use csp.py, search.py & utils.py from https://github.com/aimacode/aima-python.