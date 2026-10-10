# Deployment: Linux ARM64 / NVIDIA H200

Target architecture: Linux `aarch64`, NVIDIA H200 GPUs (cluster details TBD). **No ARM/H200 run has been performed.**

## Strategy

1. Keep numerical research logic architecture-neutral in Python.
2. Use aarch64-native installations of PyTorch selected for the cluster's CUDA driver/toolkit. Do not use the authors' x86_64-specific frozen Conda environment or blindly force x86 CUDA wheels.
3. Deploy all compiled components (GROMACS, tblite, MPI, any MiMiC/CPMD integrations) from validated native packages or compiler builds. Confirm availability on the cluster's modules.
4. Use SLURM job arrays for independent ligand/phase/seed runs, and retain target potential evaluation logs. H200 availability does not mean tblite or GROMACS will automatically run on GPU.
5. Store large datasets on shared scratch and export read-only references into jobs; keep `data/` and `runs/` out of git.
6. Record `uname -m`, `nvidia-smi`, torch CUDA capability, OS, compiler, CUDA, MPI, and upstream SHAs with each experiment.

## SLURM template

See `scripts/hpc_smoke.sbatch`. This deliberately runs only diagnostic code; adapt partition, account, environments, and modules to the actual cluster before submission.

## Readiness gates

- [ ] Native aarch64 Python and TFEP dependency installation
- [ ] PyTorch recognizes H200 and tensor/autograd smoke tests pass
- [ ] GROMACS and tblite execute consistent energy/force test cases
- [ ] TFEP upstream training run matches reference on a small dataset
- [ ] Full benchmark launched only after passing all gates
