import csp
import time
import random
from utils import argmin_random_tie, count

# -------------------- Heuristics and Propagation --------------------
# Functions i changed from csp.py

def AC3(csp_obj, queue=None, removals=None, arc_heuristic=csp.dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp_obj.variables for Xk in csp_obj.neighbors[Xi]}
    csp_obj.support_pruning()
    queue = arc_heuristic(csp_obj, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp_obj, Xi, Xj, removals, checks)
        if revised:
            if not csp_obj.curr_domains[Xi]:
                return False, checks
            for Xk in csp_obj.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks

def revise(csp_obj, Xi, Xj, removals, checks=0):
    revised = False
    for x in csp_obj.curr_domains[Xi][:]:
        conflict = True
        for y in csp_obj.curr_domains[Xj]:
            if not csp_obj.constraint(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp_obj.prune(Xi, x, removals)
            revised = True
    if not csp_obj.curr_domains[Xi]:
        csp_obj.weight[Xj] += 1
        csp_obj.weight[Xi] += 1
    return revised, checks

def dom_wdeg(assignment, csp_obj):
    return argmin_random_tie([v for v in csp_obj.variables if v not in assignment],
                             key=lambda var: domain_weight_ratio(csp_obj, var, assignment))

def domain_weight_ratio(csp_obj, var, assignment):
    if csp_obj.curr_domains is not None:
        return len(csp_obj.curr_domains[var]) / csp_obj.weight[var]
    else:
        return count(csp_obj.nconflicts(var, val, assignment) == 0 for val in csp_obj.domains[var])

def unordered_domain_values(var, assignment, csp_obj):
    return csp_obj.choices(var)

def lcv(var, assignment, csp_obj):
    return sorted(csp_obj.choices(var), key=lambda val: csp_obj.nconflicts(var, val, assignment))

def no_inference(csp_obj, var, value, assignment, removals):
    return True

def forward_checking(csp_obj, var, value, assignment, removals):
    csp_obj.support_pruning()
    for B in csp_obj.neighbors[var]:
        if B not in assignment:
            for b in csp_obj.curr_domains[B][:]:
                conflict = csp_obj.constraint(var, value, B, b)
                if conflict:
                    csp_obj.prune(B, b, removals)
            if not csp_obj.curr_domains[B]:
                csp_obj.weight[var] += 1
                csp_obj.weight[B] += 1
                return False
    return True

def mac(csp_obj, var, value, assignment, removals, constraint_propagation=AC3):
    return constraint_propagation(csp_obj, {(X, var) for X in csp_obj.neighbors[var]}, removals)[0]

# -------------------- Backtracking & Hybrid Search --------------------

def backtracking_search(csp_obj, select_unassigned_variable=csp.first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):

    def backtrack(assignment):
        if len(assignment) == len(csp_obj.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp_obj)
        for value in order_domain_values(var, assignment, csp_obj):
            if 0 == csp_obj.nconflicts(var, value, assignment):
                csp_obj.assign(var, value, assignment)
                removals = csp_obj.suppose(var, value)
                if inference(csp_obj, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp_obj.restore(removals)
        csp_obj.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp_obj.goal_test(result)
    return result, csp_obj.constraintCount, csp_obj.nassigns

def hybrid_search(csp_obj, select_unassigned_variable=dom_wdeg,
                  order_domain_values=unordered_domain_values, inference=no_inference):

    visited = set()
    for var in csp_obj.variables:
        csp_obj.conflict_set[var] = set()
        csp_obj.order[var] = 0

    def fccbj(assignment):
        if len(assignment) == len(csp_obj.variables):
            return assignment, None
        var = select_unassigned_variable(assignment, csp_obj)
        csp_obj.order[var] = csp_obj.counter
        csp_obj.counter += 1
        for value in order_domain_values(var, assignment, csp_obj):
            if 0 == csp_obj.nconflicts(var, value, assignment):
                csp_obj.assign(var, value, assignment)
                removals = csp_obj.suppose(var, value)
                if inference(csp_obj, var, value, assignment, removals):
                    result, last = fccbj(assignment)
                    if result is not None:
                        return result, None
                    elif var in visited and var != last:
                        csp_obj.conflict_set[var].clear()
                        visited.discard(var)
                        csp_obj.restore(removals)
                        csp_obj.unassign(var, assignment)
                        return None, last
                csp_obj.restore(removals)
        last = None
        biggest = 0
        csp_obj.unassign(var, assignment)
        visited.add(var)
        if len(csp_obj.conflict_set[var]):
            for conflict in csp_obj.conflict_set[var]:
                if csp_obj.order[conflict] > biggest:
                    biggest = csp_obj.order[conflict]
                    last = conflict
            csp_obj.conflict_set[last].update(csp_obj.conflict_set[var])
            csp_obj.conflict_set[last].discard(last)
        return None, last

    result, last = fccbj({})
    assert result is None or csp_obj.goal_test(result)
    return result, csp_obj.constraintCount, csp_obj.nassigns

# -------------------- Min-Conflicts --------------------

def min_conflicts(csp_obj, max_steps=1000):
    csp_obj.current = current = {}
    conf = 0
    for var in csp_obj.variables:
        val = min_conflicts_value(csp_obj, var, current)
        csp_obj.assign(var, val, current)
    for i in range(max_steps):
        conflicted = csp_obj.conflicted_vars(current)
        if not conflicted:
            return current, csp_obj.constraintCount, csp_obj.nassigns
        if i == max_steps - 1:
            conf = len(conflicted)
        var = random.choice(conflicted)
        val = min_conflicts_value(csp_obj, var, current)
        csp_obj.assign(var, val, current)
    print("Number of conflicts: ", conf)
    return None, csp_obj.constraintCount, csp_obj.nassigns

def min_conflicts_value(csp_obj, var, current):
    return argmin_random_tie(csp_obj.domains[var], key=lambda val: csp_obj.nconflicts(var, val, current))

# -------------------- File Input & Solver --------------------

constraints = {}
domains = []
variables = []
neighbors = []
ConFull = []

with open("instances/dom2-f25.txt", "r") as f:
    f.readline()
    temp = [[int(j) for j in line.split()[1:]] for line in f]

with open("instances/var2-f25.txt", "r") as f:
    num_vars = int(f.readline()) 
    neighbors = [[] for _ in range(num_vars)]
    for line in f:
        var, dom_index = map(int, line.split())
        variables.append(var)
        domains.append(temp[dom_index])


with open("instances/ctr2-f25.txt", "r") as f:
    f.readline()
    for line in f:
        var1, var2, op, res = line.split()
        var1, var2, res = int(var1), int(var2), int(res)
        if var1 not in neighbors[var2]:
            neighbors[var2].append(var1)
        if var2 not in neighbors[var1]:
            neighbors[var1].append(var2)
        ConFull.append([var1, var2, op, res])
        constraints[(var1, var2)] = [op, res]

RLFA = csp.CSP(variables, domains, neighbors, constraints, ConFull)

print("Which search method would you like to use?")
print("1. Forward Checking")
print("2. Maintaining Arc Consistency")
print("3. Forward Checking with Conflict Directed Backjumping")
print("4. Min-Conflicts")
try:
    choice = int(input("Enter your choice: "))
except ValueError:
    print("Invalid input.")
    exit()

if choice == 1:
    print("Running FC:")
    start = time.time()
    result, ncon, nassigns = backtracking_search(RLFA, select_unassigned_variable=dom_wdeg, inference=forward_checking)
elif choice == 2:
    print("Running MAC:")
    start = time.time()
    result, ncon, nassigns = backtracking_search(RLFA, select_unassigned_variable=dom_wdeg, inference=mac)
elif choice == 3:
    print("Running FC-CBJ:")
    start = time.time()
    result, ncon, nassigns = hybrid_search(RLFA, select_unassigned_variable=dom_wdeg, inference=forward_checking)
elif choice == 4:
    print("Running Min-Conflicts:")
    start = time.time()
    result, ncon, nassigns = min_conflicts(RLFA)
else:
    print("Invalid choice")
    exit()

end = time.time()
if result:
    print("Solution found")
else:
    print("No solution found")
print("Number of constraints checked:", ncon)
print("Number of assignments:", nassigns)
print("Time taken:", end - start)
