# Geomechanics

Geomechanical models couple flow (and optionally heat transport) with the deformation of the rock
on unstructured meshes. They are built on `THMCModel` with the `Poroelasticity` physics and the
`UnstructReservoirMech` reservoir, and are selected through `InputData(type_mech=...)`:
`'poroelasticity'`, `'thermoporoelasticity'`, or `'none'` for flow only.

## Formulation

The mechanics is linear (thermo-)poroelasticity, solved fully coupled and fully implicitly with
flow. For every cell the momentum balance is added to the mass (and energy) balances; in its
default, quasi-static form (the inertial one is described in *Quasi-static and dynamic formulations*
below) it reads:

```
div( C : eps(u) - B p - A T ) + rho_b g = 0                 (A T only in thermoporoelasticity)
```

with displacement `u`, small-strain tensor `eps(u) = (grad u + grad u^T) / 2`, stiffness tensor
`C`, Biot tensor `B`, thermal tensor `A` (`rock.th_expn`) and bulk density `rho_b`; pressure and
temperature are taken relative to the reference state. Stresses are
effective-stress based, in bar, and negative in compression.

Deformation enters the mass balance through the porosity:

```
phi = phi_0 + B : (eps - eps_ref) + c_s (p - p_ref) - a_phi (T - T_ref)
c_s = (b - phi_0) (1 - b) / K_d,   b = tr(B) / 3,   K_d = E / (3 (1 - 2 nu))
```

where `eps_ref`, `p_ref` and `T_ref` describe the reference (initial, equilibrated) state and
`a_phi` is `rock.th_expn_poro`.
`darts.reservoirs.unstruct_reservoir_mech` has helpers for these relations: `get_lambda_mu()`,
`get_bulk_modulus()`, `get_biot_modulus()`, `get_rock_compressibility()`,
`get_isotropic_stiffness()` and `E_nu_from_Vp_Vs()`.

### Unknowns and units

| Unknowns per cell | Poroelasticity | Thermoporoelasticity |
|---|---|---|
| `mech_discretizer` | `p`, `z_1..z_(nc-1)`, `u_x`, `u_y`, `u_z` | `p`, `z_1..z_(nc-1)`, `T`, `u_x`, `u_y`, `u_z` |
| `pm_discretizer` | `u_x`, `u_y`, `u_z`, `p` | not supported |

Displacements are in m, pressures and stresses in bar, Young's modulus `E` in bar and
compressibilities in 1/bar.

### Rock properties

Per region (`InputData.rock`):

* stiffness: isotropic `E` and `nu`, or a full 6x6 `stiffness` tensor (anisotropic rock);
* `biot`: scalar or 3x3 tensor;
* permeability (scalar, `permx/permy/permz` or a 3x3 tensor) and porosity;
* thermoporoelasticity: `th_expn` (scalar or 3x3 tensor `A` of the momentum balance, a
  thermal-stress coefficient in bar/K: the expansion coefficient times the drained bulk modulus,
  as in the `bai` case of `1ph_1comp_poroelastic_analytics`), `th_expn_poro` (the thermal term of
  the porosity) and `thermal_conductivity`.

Heterogeneous fields are assigned per cell through `init_heterogeneous_properties()`.

## Quasi-static and dynamic formulations

**Quasi-static** (default, both discretizers). The momentum balance has no inertia: displacements
respond instantly to pressure and temperature changes, and the timestep is limited only by flow,
heat transport and the nonlinear convergence. This is the right choice for depletion, injection,
subsidence and slow (aseismic) fault slip, with timesteps from hours to years.

**Dynamic** (`pm_discretizer` only). `engine_pm_cpu` adds an inertia term to the momentum balance:

```
rho d2u/dt2 = div( C : eps(u) - B p ) + rho_b g
```

discretized implicitly with a backward difference over the last two timesteps:

```
rho V [ (u - u_n) / dt  -  (u_n - u_(n-1)) / dt_(n-1) ] / dt
```

It is switched on by setting `engine.momentum_inertia` to the rock density `rho` in kg/m3 (the
default `0` gives the quasi-static form); the conversion to the bar/day units of the simulator is
done inside the engine. With inertia on, the engine also reports a mechanical CFL number in
`CFL_max`. The inertia term is skipped while the initial equilibrium is computed
(`find_equilibrium`).

The dynamic form resolves wave propagation and the fast stage of fault slip (seismic events), and
needs timesteps of the order of milliseconds or less. The two forms can be combined in one run:
quasi-static stepping up to the onset of slip, then dynamic stepping while it lasts.
`displaced_fault_reactivation` (`config['mode'] = 'mixed'`) does this from its driver loop: when the
timestep has been cut below about 1e-2 s, it sets `momentum_inertia = 2406.0`, fixes the timestep at
5e-4 s and tightens the linear-solver tolerance with `linear_solver.update_solver(...)`;
`'quasi_static'` keeps inertia off throughout.

## Discretizers

Both discretizers are cell-centred finite-volume schemes with multi-point flux and stress
approximations (MPFA for flow, MPSA for mechanics) that couple pressure and displacement in one
local problem per face. The model chooses one through `discretizer_name` (passed to
`Poroelasticity(discretizer=...)` and `UnstructReservoirMech(discretizer=...)`); each comes with
its own engine.

| | `mech_discretizer` (default) | `pm_discretizer` |
|---|---|---|
| Implementation | `discretizer/` (C++ discretizer package) | `engines/src/mech/` |
| Engine | `engine_super_elastic_cpu<NC, NP, THERMAL>` | `engine_pm_cpu` |
| Flow physics | multicomponent, multiphase OBL (as the super engine), with gravity, diffusion, kinetic reactions and heat conduction | single phase, single component |
| Thermoporoelasticity | yes | no |
| Faults with contact mechanics | no | yes (friction, slip, see `displaced_fault_reactivation`) |
| Inertia (dynamic mode) | no | yes (`momentum_inertia`) |
| Stabilization | -- | optional: `set_scheme_pm_discretizer('stabilized')` (eigen-splitting) |
| Default linear solver (open-source build) | SuperLU, through `params.linear_type` | SuperLU, through `engine.ls_params` |
| Examples | `SPE10_mech`, `THM_vs_nucleus_of_strain`, `1ph_1comp_poroelastic_*` | `displaced_fault_reactivation`, `1ph_1comp_poroelastic_*` |

`1ph_1comp_poroelastic_analytics` compares both discretizers against analytical solutions of the
Terzaghi, two-layer Terzaghi, Mandel and (thermoporoelastic) Bai problems; its default run uses
`mech_discretizer`. `1ph_1comp_poroelastic_convergence` measures convergence under mesh refinement
for both discretizers on rectangular and tetrahedral meshes, poroelastic and (with
`mech_discretizer`) thermoporoelastic.

Mechanics engines take their linear solver from `params.linear_type` / `engine.ls_params`, not from
`self.linear_solver.spec`; FS-CPR (`FSCPRSolverSpec`) is the iterative alternative to the default
direct solver, see [Nonlinear and linear solvers](solvers.md). Nonlinear iterations run in
`MechanicsNewtonSolver`. Geomechanical models are CPU-only and do not support OpenMP: `THMCModel`
stops if more than one thread is configured.

## Boundary conditions

Boundary conditions of a geomechanical model (`UnstructReservoirMech`) are assigned per boundary
tag of the mesh, using the predefined types in its `bc_type` attribute. Each boundary carries a
flow condition, a mechanical condition and, in thermal models, a temperature condition.

Every condition has the general form `a*x + b*f = r`: `a=1, b=0` is a Dirichlet condition
(prescribed value `x`), `a=0, b=1` is a Neumann condition (prescribed flux or traction `f`).

### Mechanical boundary conditions

The mechanical condition is applied separately to the normal (`n`) and tangential (`t`) components
of the boundary face. Dirichlet prescribes displacements [m], Neumann prescribes loads [bar].

| Type | Normal component | Tangential component | Meaning |
|---|---|---|---|
| `ROLLER` | `u_n = 0` | `s_t = 0` | slides along the boundary, cannot move away from it |
| `STUCK_ROLLER(un)` | `u_n = un` | `s_t = 0` | as `ROLLER`, with a prescribed normal displacement |
| `STUCK(un, ut)` | `u_n = un` | `u_t = ut` | prescribed normal and tangential displacements |
| `FREE` | `s_n = 0` | `s_t = 0` | traction-free boundary |
| `LOAD(Fn, Ft)` | `s_n = Fn` | `s_t = Ft` | prescribed normal and tangential load |
| `STUCK_T_LOAD_N(Fn, ut)` | `s_n = Fn` | `u_t = ut` | prescribed normal load and tangential displacement (no shearing with `ut = 0`) |

`u` is displacement and `s` is stress; `ut` and `Ft` are 3-component vectors.

### Flow boundary conditions

| Type | Condition |
|---|---|
| `NO_FLOW` | zero flux |
| `AQUIFER(p)` | prescribed pressure `p` |
| `FLOW(q)` | prescribed flux `q` |

### Temperature boundary conditions

In thermal models, the temperature condition uses the same Dirichlet/Neumann form: a prescribed
temperature or a prescribed heat flux.
