# Known limitations

* Platforms, configurations and parallelization
  * MPFA discretizations, for flow models and for geomechanical models (MPFA-MPSA), are not supported in the multithreaded (OpenMP) and GPU configurations.
  * The SuperLU direct linear solver is not parallelized ([issue 40](https://gitlab.com/open-darts/open-darts/-/issues/40)).
  * GPU builds are supported only on Linux (on Windows, they can run under WSL).
  * macOS is not supported.
  * The ARM64 platform is not officially supported.

* Linear solvers (see [Nonlinear and linear solvers](../technical_reference/solvers.md) for the full picture)
  * The open-source build has exactly two families of linear solvers, and they cannot be mixed:
    * **HYPRE-based**: the whole solve (matrix storage, Krylov driver and preconditioner) runs inside HYPRE.
      It is available only through `MGRSolverSpec`.
    * **open-DARTS implementation**: the in-tree solvers (`GMRESSolverSpec`, `CPRSolverSpec`, `FSCPRSolverSpec`,
      `SuperLUSolverSpec`, ...), which use their own matrix storage. Inside CPR and FS-CPR, HYPRE BoomerAMG is used
      only as the pressure-stage component.
  * The families cannot be mixed in time: a run cannot switch from a HYPRE solver to an in-tree one and back.
  * Nor can they be mixed within one matrix storage / solver / preconditioner combination. An in-tree GMRES with
    a HYPRE MGR preconditioner is not supported: `GMRESSolverSpec(prec=MGRSolverSpec(...))` is rejected with a
    `ValueError` (MGR already embeds its own FlexGMRES, so assign it directly). An MGR cycle preconditioned by an
    in-tree component is not supported either.
  * `AdaptiveSolverSpec` accepts only engine-resident CPU registry specs as candidates. Passing a `GPUSolverSpec`
    or a Python-resident solver (`PETScSolverSpec`, `PardisoSolverSpec`) raises a `TypeError` at construction.

* Mesh
  * A mesh must contain at least two cells, so that it has at least one cell-to-cell connection; a single-cell mesh is not supported.
  * Unstructured meshes are supported in Gmsh format 2.1 only; format 4 is not supported (a limitation of thirdparty/MeshIO).
