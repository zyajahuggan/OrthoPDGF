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
import multiprocessing
import time
import os

# Lock to ensure safe writing to CSV in multiprocessing
lock = multiprocessing.Lock()

# Clean the input PDB file (optional)
# cleanATOM("../structures/3mjg.pdb")
pose = pose_from_pdb("../structures/3mjg.clean.pdb")

# Define function for PDB to pose mapping
def pdbtopose(pose, pdb_number):
    mapping = {}
    for chain, resnum in pdb_number:
        get_poseres = pose.pdb_info().pdb2pose(chain,resnum)
        mapping[get_poseres] = (chain, resnum)  # Store PDB number
    return mapping 

# Define mutation scan positions
#nums1 = [('A', i) for i in range(25, 41)]
#nums2 = [('A', i) for i in range(74, 79)]
#nums3 = [('A', i) for i in range(81, 85)]
#nums = nums1 + nums2 + nums3
nums = [('A',i) for i in range(6,103)]
pdbandpose = pdbtopose(pose, nums)
positions = list(pdbandpose.keys())

# Prepare residue selector
positions_str = ",".join(map(str, positions))
selector = rosetta.core.select.residue_selector.ResidueIndexSelector(positions_str)

# Get the score function
scorefxn = get_score_function()

# Create task factory for packing
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
mmf.add_chi_action(mm_enable, selector)
mmf.all_bb(False)
mmf.all_jumps(False)

# Initialize the relax mover
relax = FastRelax(standard_repeats=1)
relax.set_scorefxn(scorefxn)
relax.max_iter(1000)
relax.dualspace(False)
relax.set_task_factory(tf)
relax.cartesian(False)
relax.set_movemap_factory(mmf)

# Mutation list
mutations = [
    "A", "C", "D", "E", "F", 
    "G", "H", "I", "K", "L", 
    "M", "N", "P", "Q", "R", 
    "S", "T", "V", "W", "Y"
]

# Define the worker function for mutation and relaxation
def mutate_and_relax(args):
    pos, wt_res, mut = args
    pyrosetta.init()  # Ensure PyRosetta is initialized in each worker process

    pose = pose_from_pdb("../structures/3mjg.clean.pdb")  # Load a fresh pose for each worker
    mut_pose = pose.clone()
    
    mutate_residue(mut_pose, pos, mut)
    relax.apply(mut_pose)
    
    # Save the mutated structure
    pdb_filename = f"../structures/pdgf_{wt_res}{pos}{mut}.pdb"
    mut_pose.dump_pdb(pdb_filename)
    
    # Compute ∆∆G
    ddg = scorefxn(mut_pose) - scorefxn(pose)
    
    # Get PDB numbering
    chain, pdb_resnum = pdbandpose[pos]
    
    # Define the full path for the CSV file
    output_dir = "/scratch4/jgray21/zhuggan1/projects/orthosystems/PDGFR/out"
    csv_filename = os.path.join(output_dir, "mutation_ddg_results_wholeligandA.csv")
    
    # Write the result to CSV immediately (using lock)
    with lock:  # Ensure safe writing in parallel processing
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([chain, pdb_resnum, wt_res, mut, ddg])

    return (pos, wt_res, mut, ddg)

# Main execution
if __name__ == "__main__":
    # Create the CSV file and write headers before running mutations
    output_dir = "/scratch4/jgray21/zhuggan1/projects/orthosystems/PDGFR/out"
    csv_filename = os.path.join(output_dir, "mutation_ddg_results.csv")
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["PDB Chain", "PDB Residue", "Wild-Type", "Mutation", "∆∆G"])

    # Prepare mutation tasks
    tasks = []
    for pos in positions:
        wt_res = pose.residue(pos).name1()  # Get the wild-type residue at this position
        if wt_res == 'C':
            continue
        for mut in mutations:
            if mut != wt_res:  # Ensure we are not mutating to the same residue
                tasks.append((pos, wt_res, mut))

    # Multi-process execution
    start_time = time.perf_counter()
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results_parallel = pool.map(mutate_and_relax, tasks)
    multi_time = time.perf_counter() - start_time    
    print(f"Multi-process execution time: {multi_time:.2f} seconds")
