# Features overview

open-DARTS is a Python-driven simulator with C++/CUDA engines. A model is a Python class that
derives from `DartsModel` and describes the reservoir, the physics, the wells and the solvers.
Below is a short map of what is available; the [example models](example_models.md) in `models/`
show each feature in use.

## Physics

All physics use Operator-Based Linearization (OBL): the nonlinear terms of the conservation
equations are tabulated on a parameter-space grid and interpolated, so any property model that
can be evaluated pointwise can be plugged in.

* **Dead oil and black oil**: `DeadOil`, `BlackOil` (`2ph_do`, `3ph_do`, `3ph_bo`).
* **Compositional**: `PhysicsBase` with property correlations (`2ph_comp`, `2ph_constant_k`,
  `3ph_comp_w`), or `EoSPhysics` with an equation-of-state flash from open-darts-flash (`CCS`, `SPE11b`).
* **Thermal and geothermal**: P-T, P-H and P-S state specifications; `IAPWSPhysics` for water
  and steam (`2ph_geothermal`, `GeoRising`, `CoaxWell`, `cpg_sloping_fault`, `fracture_network`).
* **Reactive transport**: kinetic and equilibrium reactions, precipitation/dissolution with
  a solid phase (`2ph_comp_solid`, `Chem_benchmark_new`), element-based reactive flow coupled to
  PHREEQC or Reaktoro (`chemistry`).
* **Transport processes**: advection with gravity and capillarity, molecular diffusion, heat
  conduction, and velocity-dependent dispersion of mass and heat; the cell velocities that
  dispersion needs are reconstructed from the fluxes (`reconstruct_velocities()`).
* **Special effects**: relative-permeability and capillary-pressure hysteresis
  (Killough/Land, `2ph_hysteresis`), foam (`3ph_comp_w_foam`, `CO2_foam_CCS`),
  specific potential energy in the energy equation (`effect_of_potential_energy`).
* **Property library** (`darts.physics.properties`): Brooks-Corey and Van Genuchten relative
  permeability and capillary pressure, density/viscosity/enthalpy/conductivity correlations,
  flash models (constant K, single phase, solid, ionic), kinetic rate laws.

## Geomechanics

* **Poroelasticity and thermo-poroelasticity** on unstructured meshes (`THMCModel`,
  `Poroelasticity`), validated against analytical solutions (`1ph_1comp_poroelastic_analytics`,
  `1ph_1comp_poroelastic_convergence`).
* **Faults with contact mechanics**: frictional slip and fault reactivation
  (`displaced_fault_reactivation`).
* **Boundary conditions**: prescribed displacements, loads, rollers and free faces on the mechanical
  boundaries; pressure, flux or no-flow and temperature conditions on the flow boundaries.
* **Semi-analytical nucleus-of-strain** displacements and stress estimates (`THM_vs_nucleus_of_strain`).

See [Geomechanics](../technical_reference/geomechanics.md) for details.

## Reservoir description and meshes

* **Structured** Cartesian grids (`StructReservoir`).
* **Radial** grids around a well: structured 1D/2D (r-z) grids with optional logarithmically growing
  radial cells (`StructRadialReservoir`, used in `CCS` and `dfm_well`), and unstructured radial meshes
  (`UnstructRadialReservoir`).
* **Corner-point geometry** (`CPG_Reservoir`): GRDECL input, faults transmissibility multipliers, generation of simple structured-like CPG grids,
  export to VTK (`cpg_sloping_fault`).
* **Unstructured** meshes from Gmsh (`.msh`, format 2.1) with TPFA or MPFA discretization
  (`UnstructReservoir`, `2ph_do_thermal_mpfa`, `Uniform_Brugge`, `SPE11b`).
* **Discrete fracture networks**: fracture preprocessing, intersection and meshing tools in
  `darts.tools.fracture_network` (`fracture_network`).

See [Reservoir](../technical_reference/reservoir.md) for details.

## Wells

* Wells with BHP and molar, mass, volumetric or advective-heat rate controls and
  constraints, which may change in time.
* **Drift-flux wellbore model** (`DFMWellModel`, `darts.pipes`) for multiphase flow in the
  wellbore coupled to the reservoir (`dfm_well`).

See [Wells](../technical_reference/wells.md) for details.

## Initial conditions

* Uniform values, per-cell arrays (`set_initial_conditions_from_array()`), or a depth table
  interpolated to the cell depths (`set_initial_conditions_from_depth_table()`), e.g. for
  hydrostatic or geothermal-gradient initialization.

## Boundary conditions

* **Flow models**: closed boundaries by default; constant-pressure boundaries by assigning large
  volumes to boundary cells (`set_boundary_volume()`); custom sources and boundary fluxes by
  overriding `set_rhs_flux()`, which returns per-cell rates that are added to the residual in
  every Newton iteration.
* **Geomechanical models**: see [Geomechanics](../technical_reference/geomechanics.md).

## Solvers and time stepping

* Newton solver with update chopping, OBL-box clamping, divergence guards and adaptive
  time stepping with regrowth control.
* Linear solvers: in-tree FGMRES with CPR / FS-CPR (HYPRE BoomerAMG pressure stage), HYPRE MGR,
  SuperLU, PETSc and Pardiso on CPU; AMGX-CPR, cuSPARSE and cuDSS on GPU; Schur elimination and
  adaptive switching between solvers.

See [Nonlinear and linear solvers](../technical_reference/solvers.md) for details.

## Adjoint gradients and history matching

Adjoint gradients of well-data misfit with respect to transmissibilities and well indices, for
history matching and optimization (`Adjoint_super_engine`, `Adjoint_mpfa`,
`Adjoint_PXflash_geothermal`).

## Performance

* Serial, multithreaded (OpenMP) and GPU (CUDA, Linux only) builds; see
  [Configure hardware](configure_hardware.md).
* Adaptive OBL: supporting points are evaluated on demand, can be cached on disk and reused
  between runs, and can be computed in parallel worker processes (`ParallelEvaluator`).

## Input

* Models are set up in Python: a `DartsModel` subclass defines the reservoir, physics, wells,
  initial and boundary conditions, and solvers.
* Input data can be grouped in the `darts.input` data classes (`InputData` with rock, fluid,
  initial-solution, well, OBL and simulation parameters), from which wells and their controls
  are set up by `set_well_controls_idata()`.
* Grid and property arrays can be read from Eclipse-style keyword files (GRDECL: `SPECGRID`,
  `COORD`, `ZCORN`, `ACTNUM`, `PERMX`, `PORO`, ..., also gzip-compressed); only a subset of the
  keywords is supported.
* Unstructured meshes are read from Gmsh `.msh` files (format 2.1); `darts.tools` also has helpers
  to generate box meshes and to convert VTU files to `.msh`.

## Output

* Results are written to HDF5 files in the output folder (`set_output()`): the reservoir state
  (`reservoir_solution.h5`) at the end of every `run()` call, and the well-block state
  (`well_data.h5`) at every converged timestep.
* Only the primary variables are stored. Secondary properties (saturations, densities, phase
  compositions, ...) are evaluated in a **post-processing mode**: `output.output_properties()`
  reads the saved states (or the current engine state) and evaluates the requested properties
  with the model's property containers, so no property output has to be planned before the run.
* The same evaluation feeds the exporters: VTK for ParaView (`output_to_vtk()`, plus VTP for well
  trajectories), xarray datasets (`output_to_xarray()`) and quick matplotlib slices
  (`output_to_plt()`).
* Well time series (BHP, temperature, phase and component rates at perforations and well heads)
  are computed from the well data by `output.store_well_time_data()`, saved to `.pkl` and
  `.xlsx`, and plotted by `output.plot_well_time_data()`.

## Restart

* `load_restart_data(filepath, ts_idx=-1)` restarts a model from a saved reservoir solution
  (`reservoir_solution.h5`), by default from its last saved state; `ts_idx` selects another one.


