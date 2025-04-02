#!/usr/bin/python3

#interact -a jgray21
#source /scratch4/jgray21/zhuggan1/miniconda3/etc/profile.d/conda.sh
#conda activate /scratch4/jgray21/zhuggan1/envs/envs/pyrosetta

import pyrosetta
pyrosetta.init()
from pyrosetta import *
from pyrosetta.toolbox import mutate_residue
from rosetta.protocols.relax import FastRelax
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.core.select.movemap import *
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.pack.task.operation import OperateOnResidueSubset, PreventRepackingRLT
from pyrosetta.rosetta.utility import *
from pyrosetta.rosetta.core.scoring import hbonds
import sys
import csv

# Clean the input PDB file
cleanATOM("3mjg.pdb")
pose = pose_from_pdb("3mjg.clean.pdb")

# Defining interaction sphere
pose_nums = [pose_res for pose_res in range(1, pose.total_residue() + 1)]
positions = pose_nums[24:40] + pose_nums[73:77] + pose_nums[80:84]
positions_str = ",".join(map(str, positions))
selector = rosetta.core.select.residue_selector.ResidueIndexSelector(positions_str)

# Get the score function
scorefxn = get_score_function()

# Create a TaskFactory and use RestrictToRepacking to prevent amino acid changes
tf = rosetta.core.pack.task.TaskFactory()
prevent_repacking = PreventRepackingRLT()
prevent_subset_repacking = rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking, selector, True)
tf.push_back(prevent_subset_repacking)
tf.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())
tf.push_back(rosetta.core.pack.task.operation.InitializeFromCommandline())
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

# Specifies which residues are allowed to move
mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
mmf.add_bb_action(mm_enable, selector)
mmf.add_chi_action(mm_enable, selector)

# Initialize the relax mover
relax = FastRelax(standard_repeats=1)
relax.set_scorefxn(scorefxn)
relax.max_iter(1)
relax.dualspace(False)
relax.set_task_factory(tf)
relax.cartesian(False)
relax.set_movemap_factory(mmf)

### Define mutation scan parameters ###
mutations = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

results = []
print('Dumping initial structure...')

# Perform mutations
for pos in positions:
    wt_res = pose.residue(pos).name1()
    for mut in mutations:
        if mut == wt_res:
            continue
        print(f'Checking mutation {mut} at position {pos}')
        
        mut_pose = pose.clone()
        mutate_residue(mut_pose, pos, mut)
        print('Next Mutation')
        
        # Relax structure
        relax.apply(mut_pose)
        
        # Calculate ∆∆G
        ddg = scorefxn(mut_pose) - scorefxn(pose)
        results.append((pos, wt_res, mut, ddg))

# Sort by ∆∆G
results.sort(key=lambda x: x[3])

# Save results to CSV file
csv_filename = "mutation_ddg_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Position", "Wild-Type", "Mutation", "∆∆G"])
    writer.writerows(results)

print(f"Results saved to {csv_filename}")

# Print top stabilizing mutations
print("\nTop Stabilizing Mutations:")
for res in results[:10]:
    print(f"Pos {res[0]}: {res[1]} → {res[2]}, ∆∆G = {res[3]:.2f}")

# Print most destabilizing mutations
print("\nMost Destabilizing Mutations:")
for res in results[-10:]:
    print(f"Pos {res[0]}: {res[1]} → {res[2]}, ∆∆G = {res[3]:.2f}")

