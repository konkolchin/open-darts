# THM vs Geomechanical Proxy

Comparison of a fully-coupled **Thermo-Hydro-Mechanical (THM)** reservoir simulation against a fast analytical **geomechanical proxy** for thermoporoelastic responses.

The THM simulation runs inside [open-DARTS](https://gitlab.com/open-porous-media/open-darts) on a **3D** unstructured hexahedral mesh. The proxy evaluates the same displacements and stresses analytically from the pressure/temperature changes produced by the THM solver, making it orders of magnitude faster while remaining accurate for the far-field mechanical response.

Both the THM solver and the proxy are verified against the closed-form solution for a **laterally infinite** (uniaxial-strain) reservoir layer.

**Poroelastic horizontal total stress change**:

$$\Delta\sigma_H^\text{total} = \frac{\alpha(1-2\nu)}{1-\nu}\,\Delta P$$

where α is the Biot coefficient and ν is Poisson's ratio. With α = 0.7 and ν = 0.25 (the values pinned for the CI reference cases) the ratio is ≈ 0.47; with α = 1 it reduces to the often-cited value 2/3. The model default is α = 0.9 (see *Rock properties and defaults*). Vertical total stress is unchanged (Δσv = 0) because the reservoir is free to deform vertically.

**Thermoelastic horizontal total stress change**:

$$\Delta\sigma_H^\text{total} = \frac{E\,\alpha_T}{1-\nu}\,\Delta T$$

where E is Young's modulus and α_T is the linear thermal expansion coefficient (1×10⁻⁵ K⁻¹ by default). Heating produces compressive horizontal stress; cooling produces tension.

The `print_at_point` mode in `main_nucleus_of_strain.py` prints both the THM and proxy values at the reservoir mid-point alongside these limits.

---

## Overview

| Step | Tool | What it does |
|------|------|--------------|
| 1 | `main.py` | Runs the full THM simulation; writes VTK output |
| 2 | `main_nucleus_of_strain.py` | Reads THM VTK output, runs the proxy, plots comparison |

**Physics and wells are set by the case.** Each case builds on one of two layers in `cases/base.py`, which also sets the duration and the vtk output interval; a case can override any of them:

| Layer | `physics_type` | `wells_type` | `runtime_years` | `report_step` [days] |
|-------|----------------|--------------|-----------|----------------------|
| `base_HM()` | `single_phase`: isothermal poromechanics (pressure + mechanics) | `inj`: one injection well (BHP control) | 10 | 365.25/2 |
| `base_THM()` | `single_phase_thermal`: thermoporoelasticity (pressure + temperature + mechanics) | `doublet`: injection + production pair (rate control) | 60 | 365.25/2 |

All named cases (`homog`, `heter`, `damage_zone`, `case_1..4`, ...) are THM. The struct-like meshes are HM as `rect_NXxNYxNZ` and THM as `thm_rect_NXxNYxNZ`; the results folder (`results/<physics_type>_<wells_type>_<case>`) leaves the `thm_` prefix out, since the physics is already in it. `wells_type = 'prod'` (one production well) is supported by the model, but no case uses it.

---

## Repository layout

Only source files tracked in git are listed below. Generated artefacts (`results/`, `.spyproject/`) are local and not committed; generated meshes are written into the run's own output folder.

```
THM_vs_nucleus_of_strain/
├── main.py                      # Entry point: run THM simulation (+ fault post-processing, reference checks)
├── main_nucleus_of_strain.py                # Entry point: run proxy & compare vs THM
├── model.py                     # THMCModel subclass (physics, wells, IC/BC)
├── reservoir.py                 # UnstructReservoirMech subclass (mesh, VTK output, fault traction)
├── set_case.py                  # Discovers the cases in cases/*.py and builds their input data
├── plot_vtk.py                  # PyVista slice plots of the vtk output
├── fault.py                     # Fault stability: Mohr-Coulomb / FSP, slip time series, fault maps, dip profiles
├── check_equil.py               # Stress after the equilibration step vs the uniaxial-strain (K0) solution; analytic initial FSP
├── check_mesh.py                # Mesh consistency checker
├── gen_fault_msh.py             # gmsh generator: fault with a damage zone (damage_zone)
├── gen_fault_msh_no_damage_zone.py  # gmsh generator: fault without a damage zone (homog, heter, ...)
├── cases/
│   ├── base.py                  # input_data_base(), input_data_struct_like(), rock-property checks and modifiers
│   ├── case_1.py                # case_1..case_4: per-tag properties on an externally generated mesh (example)
│   ├── damage_zone.py           # damage_zone
│   ├── homog.py                 # homog and its variants (offset, dip, Biot, coupled, 1 layer, finer mesh)
│   └── heter.py                 # heter: sand reservoir / shale burden through mesh tags
└── ref/                         # CI references (vtk and well time-series pkl) of rect_17x17x15
```

The proxy itself (`nucleus_of_strain`) lives in `darts/physics/geomechanics/`; the box-mesh generator in `darts/tools/gen_msh.py`; formatting helpers in `darts/tools/logging.py`.

**C++ extension** `_proxygeomech` (source in `engines/src/mech_nucleus_of_strain`) is built with open-darts and installed into `darts/physics/geomechanics/nucleus_of_strain/`; no separate build is needed. A GPU build of open-darts adds the CUDA path to the same module.

---

## Geometry cases

Two mesh strategies are supported, selected by the case name passed to `run()` / `run_geomech_nucleus_of_strain()`.

### Rectangular hexahedra, generated on the fly (`NXxNYxNZ` naming)

Cases named as `rect_NXxNYxNZ` (HM) or `thm_rect_NXxNYxNZ` (THM) (e.g. `rect_17x17x15`, `rect_41x41x66`, `rect_71x71x66`, `rect_83x83x90`, `thm_rect_97x97x90`) use `cases/base.py::input_data_struct_like`. The mesh is built at runtime by Gmsh (`darts.tools.gen_msh`, shared with the other models) as a structured grid of regular hexahedra when `idata.other.generate_mesh` is `True` (the default); a mesh whose inputs did not change is reused.

- **Single mesh tag** (`99991`) covers the entire domain.
- Rock properties (permeability, porosity, Young's modulus) are assigned per cell by coordinate-based interpolation: reservoir-layer values inside the bounding box `[rsv_top, rsv_bottom] × [rsv_x1, rsv_x2] × [rsv_y1, rsv_y2]`, non-reservoir values elsewhere.
- Cell coordinates are refined near the reservoir and near well locations; the refinement pattern for each `NX` value is defined in `cases/base.py::_xc_from_nx`.
- Analytical-check cases (`cases/check_analytical.py`), for checking the uniaxial-strain formulas above, on the `rect_17x17x15`, `rect_41x41x66` and `rect_71x71x66` meshes (`rect_<mesh>_check_analytical` HM, `thm_rect_<mesh>_check_analytical` THM): a 1000 mD reservoir in practically impermeable rock (1e-6 mD). `_rect` sets the reservoir width: `rsv_xy = 10000` covers the whole mesh, which with the roller side boundaries is a laterally infinite layer (THM then matches the poroelastic formula); `rsv_xy = 1000` gives a 2 × 2 km reservoir, for which the nucleus-of-strain solution is comparable (see `notes/THM_stress_spike.md`). Each case sets its own duration (`idata.other.runtime_years`); the THM one also a shut-in time (`idata.other.shut_in_time`), after which the wells run at rate 0 so the pressure equilibrates.
  - `thm_rect_17x17x15_check_analytical` (thermoelastic, THM): injector at (0, 2 km), producer at (0, -2 km), 100 years of injection, 5 years shut in; the cooled zone stays.
  - `rect_17x17x15_check_analytical` (poroelastic, HM): one injector in the centre under BHP control, 10 years of injection without shut-in; the pressure change is nearly uniform.

### Pre-generated mesh with multiple tags (`case_1` ... `case_4`)

These cases use an externally generated mesh with **three Gmsh physical tags**: tag 1 = overburden, tag 2 = reservoir, tag 3 = underburden, selected with `idata.other.mesh_dir` / `mesh_root`. The repository no longer ships that mesh; `cases/case_1.py` stays as the example for plugging in a mesh made elsewhere (no local gmsh generator). Properties per tag can differ independently.

The cases are layered on top of each other:

```
cases/base.py :: base_THM()
    └── cases/case_1.py :: case_1()   ← sets rsv bounds, perm=1000 mD, matrix_tags=(1,2,3)
            └── case_2()   ← overrides props per tag (set_props_by_tags=True)
                    └── case_3()   ← also sets per-tag heat capacity and thermal conductivity
                            └── case_4()   ← also per-tag E, nu, Biot and thermal expansion
```

| Case | Property assignment | Rock property regions |
|------|---------------------|-----------------------|
| `case_1` | Coordinate interpolation (`set_props_by_tags=False`) | Two regions: reservoir (cells inside `rsv_top/bottom/xy` bounds) and surroundings |
| `case_2` | Per mesh tag (`set_props_by_tags=True`) | Three regions matching the Gmsh tags; vertical permeability is ×0.1 of horizontal |
| `case_3` | Per mesh tag | Same as case_2 + different heat capacity and thermal conductivity per tag (sand vs shale values) |
| `case_4` | Per mesh tag | Per-tag E, ν, Biot and thermal expansion as well |

Using `set_props_by_tags=True` lets each tag carry its own full rock-property set, making it straightforward to represent any number of geological layers as long as the mesh is tagged accordingly.

### Faulted meshes generated by gmsh

| Case | Generator | Geometry |
|------|-----------|----------|
| `damage_zone` | `gen_fault_msh.py` | 3D fault with a damage zone on both sides; the well planes make an angle with the fault strike. `damage_zone_mesh_<size>m` selects a far-field gmsh size [m] (the well size keeps the 1:20 ratio). |
| `homog`, `heter` | `gen_fault_msh_no_damage_zone.py` | Fault without a damage zone, 20 × 20 km, 5 km deep, extruded in layers along Y. `homog`: homogeneous properties; `heter`: sand reservoir, shale over/underburden through mesh tags. |

Variants of `homog` (and `heter` where it applies): `_offset0` (no reservoir offset across the fault), `_dip80` / `_dip89`, `_biot07`, `_coupled` (mechanics feeds back into the flow), `_1layer` (one layer along Y, plane strain) and `_1layer_lc50` (finer background mesh, lc = 50 m).

`homog_zero_rate` is `homog` with the wells shut in (initialisation check); `homog_deeper1km` puts the reservoir, the fault and the wells of `homog` 1 km deeper (`set_zero_rate` and `set_depth_shift` in `cases/base.py` can be applied to other cases the same way).

**Which generator is used** follows from the name: `rect_*` is a box mesh built in the reservoir, `damage*` uses `gen_fault_msh.py`, `case_*` loads an existing mesh, every other name uses `gen_fault_msh_no_damage_zone.py`. A new case is a function in `cases/*.py` that takes no arguments; there is no list to update.

### Mesh physical tags

The physical-group tags are defined once, in `cases/base.py::_set_mesh_tags`, and read from `idata.mesh`:

| `idata` | Default | Meaning |
|---------|---------|---------|
| `mesh.bnd_tags` | `BND_X-` 991, `BND_X+` 992, `BND_Y-` 993, `BND_Y+` 994, `BND_Z-` 995 (top), `BND_Z+` 996 (bottom) | 2D boundary surfaces; the boundary conditions look cells up by them |
| `mesh.matrix_tags` (from `idata.other.matrix_tags`) | `(99991,)`; `homog`: `(99991, 99992, 99993)` | volumes, **positional**: `gen_fault_msh_no_damage_zone` expects (reservoir, overburden, underburden), `gen_fault_msh` (reservoir, overburden, underburden, damage zone left, damage zone right, fault) |
| `mesh.fault_tag` | `9991` | fault surfaces in `mesh_fault.msh` (read by `fault.py`) |

`generate_box_3d` and both gmsh generators create the groups with these numbers, so changing a tag in a case changes the mesh (and regenerates it). An externally generated mesh must carry the same numbers; `check_mesh.py` reports a mismatch.

### Rock properties and defaults

`cases/base.py` sets the defaults: ν = 0.25, Biot = 0.9, fault dip 80°, friction 0.6, cohesion 0. The CI reference cases pin Biot = 0.7 (`input_data_base_biot_07`). Overrides of E, ν or Biot must be followed by `recompute_rock_derived(idata)` (bulk modulus, rock compressibility `(α − φ)(1 − α)/K_d`, thermal expansion). `run()` prints all rock properties (`print_idata`) and `set_input_data` checks them: `check_rock_props` rejects invalid ones (ν ≥ 0.5, porosity or Biot out of bounds). With Biot → 1 the rock compressibility vanishes and the thermal doublet stalls the linear solver; see `notes/biot_convergence.md`.

---

## Running

### 1. THM simulation only

Edit the `if __name__ == '__main__':` block in `main.py` to select a case and physics, then:

```bash
python main.py
```

The block only lists the cases (`cases`, any function in `cases/*.py`, `rect_NXxNYxNZ` / `thm_rect_NXxNYxNZ`,) and `check_ref` (`True` ignores the case list and repeats the CI reference check, `run_ref_checks()`). Everything else comes from the case, and `run()` arguments override it for one run:

| `run()` argument | Case setting (default) | Description |
|------------------|------------------------|-------------|
| `sim_time` | `idata.other.runtime_years` (HM 10, THM 60) | Simulated time [days] after the equilibration |
| `report_step` | `idata.other.report_step` (365.25/2) | VTK output interval [days] |
| `generate_mesh` | `idata.other.generate_mesh` (`True`) | Generate the mesh (a mesh whose inputs did not change is reused); `False` loads the mesh of the previous run. `case_1..4` always load their committed mesh |
| `decouple_geomech` | `idata.other.decouple_geomech` (`True`) | Disable mechanics→porosity feedback |
| `cache_discretization` | `idata.other.cache_discretization` (`False`) | `True` reuses a cached discretization (see below) |
| `plot_vtk_timesteps` | – | Timestep indices to visualise after the run (`[0, -1]` in the block; slices in `idata.other.plot_slices`, XZ and YZ by default) |

Output is written to `results/<physics_type>_<wells_type>_<case>/`.

### Fault stability (FSP) post-processing

`run()` in `main.py` does this automatically at the end (`plot_fault=True`, function `postprocess_fault`): FSP appended to the fault files, slip time series, 2D `FSP`/`delta_FSP` maps per report step and dip profiles through every well. Cases without a fault mesh skip it. The 2D slices of `plot_vtk_timesteps` pass through `idata.other.plot_slice_origin` (`None`: mesh center). In `homog*` the domain is `DOMAIN_W` × `DOMAIN_H` = 20 × 20 km (depth 5 km, 500 m extrusion layers along Y, set in `gen_fault_msh_no_damage_zone.py`) with the fault crossing mid-depth at X = `DOMAIN_W`/2. The doublet straddles the fault: injector 500 m west of it, producer 500 m east (1 km apart; `well_offset` in `cases/homog.py`), both at Y = `DOMAIN_H`/2. Both request the depth range 2000–3000 m, and perforations are restricted to the reservoir tag. Each well is therefore perforated over the whole local reservoir layer, which the fault offsets: ~2450–2750 m at the injector, ~2250–2550 m at the producer; slices go through the injector row at 2575 m depth. Both wells are on mass-rate control at 8000 m³/day (`idata.other.well_rate_m3_day`, balanced doublet, no BHP limits), and the injected water is 40 K colder than the reservoir.

For cases with a fault (`homog*`, `heter*`), `gen_fault_msh_no_damage_zone.py` writes `mesh.msh` (simulation) and `mesh_fault.msh` (same geometry with the fault surfaces tagged `idata.mesh.fault_tag`, 9991 by default). During the run, `reservoir.save_fault_traction` writes the fault surface with the total-stress traction for every report step (`fault<N>.vtu`, `fault.pvd`). The traction is taken from the engine connection forces (Hooke + Biot + thermal) on the mesh faces lying on the fault. Then run:

```bash
python fault.py results/single_phase_thermal_doublet_homog --friction 0.6 --cohesion 0
```

Per fault face, with `t` the traction, `n` the unit normal and `p` the pore pressure (`fault.fault_stability`):

```
sigma_n     = t . n                                  (compression positive)
tau         = |t - sigma_n n|
sigma_n_eff = sigma_n - p                            (full pore pressure; the fault has its own constitutive
                                                      behavior, so the matrix Biot coefficient is not used)
mcc         = tau - (cohesion + friction * sigma_n_eff)
FSP         = tau / sigma_n_eff                      (NaN where sigma_n_eff <= 0)
```

The run prints this computation step by step for the face with the largest FSP (`print_face_computation`).

This appends to every `fault<N>.vtu`: `sigma_n` (compression positive), `sigma_n_eff = sigma_n - p`, `tau`, `mcc = tau - (cohesion + friction * sigma_n_eff)` (> 0 means slip), `FSP = tau / sigma_n_eff` (NaN where `sigma_n_eff <= 0`), `delta_FSP` (change since the first step) and `slip`. It also writes `fault_slip.csv` and `fault_slip_vs_time.png` (FSP mean/max and slipping area [km², %] vs time), and 2D maps per report step `fault_plots/<field>_step<NNNN>.png` (fault faces unfolded onto the fault plane, along strike vs down dip, one color range per field over all steps). `--plot FSP mcc tau` selects the mapped fields (default `FSP`); `--plot` with no fields skips the maps. From Python: `plot_fault_case(case_dir, fields=('FSP', 'delta_FSP'))` or `plot_fault_field(vtu_filename, 'FSP')`. 1D profiles vs depth along the fault dip (pore pressure, temperature for thermal runs, normal stress total/effective, shear stress, Coulomb stress `tau - (c + mu sigma_n_eff)`, FSP) at one along-strike position, for several report steps: `--profile 4250 5750` (no value: at the face with the largest FSP), `--profile-steps 0 30`; from Python `plot_fault_dip_profiles(case_dir, strike=4250., steps=[0, 30])` or `point=[x, y, z]` (e.g. a well) instead of `strike`. Saves `fault_plots/dip_profile_strike<S>m.png` and `.csv`. To compare runs (files are only read): `compare_cases([dir1, dir2], labels=['homogeneous', 'heterogeneous'], frictions=(0.6, 1.2), png_filename='fault_slip_comparison.png')` plots slipping area and FSP max/mean vs time per case and friction.

### 2. THM + proxy comparison

Edit the `if __name__ == '__main__':` block in `main_nucleus_of_strain.py`, then:

```bash
python main_nucleus_of_strain.py
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `cases` | Same as `main.py`; the physics, wells, duration and report step come from the case |
| `run_thm` | `True` to re-run the THM solver before proxy |
| `timestep_list` | VTK timestep indices to pass through the proxy: the last report step of the case, for THM also 1 and 10 years |
| `modes` | List of analysis modes (see below) |
| `n_threads` | CPU thread count for proxy evaluation |
| `use_gpu` | `True` to use the CUDA path (needs a GPU build of open-darts) |
| `read_from_cache` | `True` to reload proxy results from `.pkl` without recomputing |

**`modes` options**

| Mode | Output |
|------|--------|
| `plot_vertic_line` | 1-D vertical profiles: displacements, strains, stresses (THM vs proxy) |
| `plot_2d_slices` | 2-D XZ contour maps: displacements, effective/total stresses |
| `plot_horiz_line` | ΔP along the X axis at fixed Y, Z, over multiple timesteps |
| `print_at_point` | Console comparison at reservoir mid-point (includes analytical reference) |
| `check_initial` | Validate initial pressure and vertical stress against density-based gradients |

HTML and PDF summary reports (`proxy_vs_thm_2d.html`, `proxy_vs_thm_report.pdf`) are saved automatically in the timestep output folder.

---

## Key physics details

### THM model (`model.py`)

- Inherits `THMCModel` from open-DARTS.
- Fluid: single phase, single component (water). No phase equilibrium or compositional flash is needed; the single component fills the pore space with saturation = 1.
- Solver: CPR-preconditioned GMRES (`cpu_gmres_fs_cpr`).
- Mechanical equilibrium is found by running a large fictitious timestep (`dt = 1e8` days) before the actual simulation starts; subsequent displacements are reported relative to that initial state.
- Wells are placed by finding the mesh cell whose centroid is closest to the requested (X, Y, Z) perforation coordinate; well index is computed using the Peaceman formula.

**Optional: temperature-dependent fluid properties** (`model.py`)
The two classes below are defined in `model.py` but are **not active by default**. They can be enabled by overriding the property containers after `set_physics()` is called (see `set_physics_dummy` in `model.py` for an example):

| Class | Replaces | What it adds |
|-------|----------|--------------|
| `MaoDuan2009Shifted` | constant viscosity | State-dependent water viscosity via the Mao-Duan 2009 correlation. Wraps `MaoDuan2009` and shifts the model's relative temperature axis (baseline 0) to absolute Kelvin by adding `t_abs0=373.15 K` (100 °C), so the correlation always receives a physical temperature. |
| `DensityBasicTdep` | `DensityBasic` | Adds a linear thermal-expansion correction: `ρ(p,T) = ρ₀ · (1 + c·(p−p₀) − β·(T−T₀))`, where `β` is the volumetric thermal expansion coefficient. |

### Reservoir (`reservoir.py`)

- Inherits `UnstructReservoirMech`.
- Boundary conditions: roller (no normal displacement) on all lateral and bottom faces; free top surface.
- Thermal BC: aquifer (constant temperature) on all boundaries.
- Rock properties are assigned either by nearest-neighbour interpolation from a structured grid or per Gmsh physical tag.
- Discretization cache (off by default): the expensive discretizer steps are stored in `<mesh_root>/<case>/.cache/discretization_*.npz` and reused while the mesh, tags, boundary conditions, rock data, gravity and the discretizer build are unchanged. A changed input gives a new file; a corrupt file falls back to discretizing; only the 2 latest files are kept. Enable with `idata.other.cache_discretization = True` or `run(..., cache_discretization=True)`.

### Geomechanical proxy (`_proxygeomech`)

The proxy solves the thermoporoelastic response analytically by integrating the Green's function for a rectangular prism source (Mindlin-type). It accepts:
- Source cell bounding boxes (prisms) and their ΔP, ΔT values from the THM VTK output.
- Evaluation point coordinates.

And returns displacements, strains, and effective/total stresses split into poroelastic and thermoelastic contributions. GPU acceleration is available in GPU builds of open-darts (`set_platform('gpu')`).

---

## Output files

All output is written under `results/<physics_type>_<wells_type>_<case>/`.

### THM simulation (`main.py`)

| File | Description |
|------|-------------|
| `solution<N>.vtu` | Per-timestep VTK unstructured grid (written by `reservoir.py`). Cell fields: `pressure` [bar], `temperature` [K], `ux`/`uy`/`uz` (displacements relative to the geomechanical equilibrium state) [m], `tot_stress` (total stress, 6-component Voigt) [bar], `eff_stress`, `delta_tot_stress`, `delta_eff_stress`, `delta_pressure`, `delta_temperature`, `strain`, `perm` (full 3×3 permeability tensor) [mD], `E` [bar], `poisson`, `poro`, `viscosity` [cP]. |
| `solution.pvd` | ParaView collection file referencing all `.vtu` timesteps. |
| `fault<N>.vtu`, `fault.pvd` | Fault surface (cases with `mesh_fault.msh` only). Cell fields: `traction` (total stress, compression positive) [bar], `normal`, `pressure` [bar], `temperature` [K] (thermal runs); after `fault.py` also `sigma_n`, `sigma_n_eff`, `tau`, `mcc` [bar], `FSP`, `slip`. |
| `fault_slip.csv`, `fault_slip_vs_time.png` | FSP (mean/max) and slipping fault area per report step, written by `fault.py`. |
| `wells.vtk` | Well tube geometry (one cylinder per perforation) for overlay in ParaView. |
| `thm_plots_timestep_<N>/<arr>_slice_<xz,yz,xy>.png` (slices through the mesh center, or through `slice_origin=[x, y, z]` passed to `plot_vtk_pyvista`, e.g. a well at reservoir depth) | XZ slice plots produced by `plot_vtk.py` for each timestep index in `plot_vtk_timesteps`. One PNG per array: `u_x,m`, `u_y,m`, `u_z,m`, `temperature,K`, `pressure,MPa`, `delta_temperature,K`, `delta_pressure,MPa`, `viscosity,cP`, `perm_XX,mD`, `perm_ZZ,mD`, `delta_eff_stress_XX/YY/ZZ,MPa`, `delta_tot_stress_XX/YY/ZZ,MPa`. Arrays absent from the VTK file (e.g. `temperature` for isothermal runs) are skipped. |
| `thm_plots_timestep_<N>/time_<t>_years.txt` | Simulation time of the figures in the folder (the folder name gives only the VTK step index). Also written to `thm_vs_nucleus_plots_timestep_<N>`. |

### Proxy comparison (`main_nucleus_of_strain.py`)

Output goes into `results/.../thm_vs_nucleus_plots_timestep_<N>/` for each processed VTK timestep index `N`; `time_<t>_years.txt` there gives the simulation time. The along-X line plots go into `results/.../plots_delta_<pressure,temperature>_x/`.

| File | Mode | Description |
|------|------|-------------|
| `mesh_skeleton.png` | always | XZ view of mesh cell boundaries with reservoir outline and well locations. |
| `delta_pressure_along_x.png` | `plot_horiz_line` | ΔP [MPa] along the X axis at Y = 0, Z = 2200 m, one curve per report step (up to 4, evenly spaced between the first and the last step of the case). |
| `delta_temperature_along_x.png` | `plot_horiz_line` | THM only: ΔT [K] along the X axis through the injector (its Y) at the reservoir mid-depth, same report steps. |
| `dp_contour.png`, `dt_contour.png` | `plot_2d_slices` | ΔP and ΔT XZ contour maps at Y = 0 interpolated from THM cell centres. |
| `dp_xy_contour.png`, `dt_xy_contour.png` | `plot_2d_slices` | ΔP and ΔT XY maps of the cell layer at the reservoir mid-depth over the whole mesh, with the wells; shows a cooled zone that the Y = 0 slice misses (e.g. wells at Y = ±2 km). |
| `<variable> - Nucleus of strain_contour.png` | `plot_2d_slices` | Proxy 2-D XZ field (displacements, effective/total stresses). |
| `<variable> - THM_contour.png` | `plot_2d_slices` | Same field from the THM VTK solution, interpolated to the same grid. |
| `<variable> - Difference_contour.png` | `plot_2d_slices` | Point-wise difference THM − Proxy on the XZ grid. |
| `<array>_<loc>_all.png` | `plot_vertic_line` | 1-D vertical profile of a field (displacement, stress, strain) at a reference XY location: THM (blue) vs proxy (red dashed). |
| `<array>_<loc>_all_diff.png` | `plot_vertic_line` | Absolute difference THM − Proxy for the same profile. |
| `proxy_vs_thm_2d.html` | `plot_2d_slices` / `plot_vertic_line` | Self-contained HTML report embedding all 2-D contour images and 1-D profile plots in a single page. |
| `proxy_vs_thm_report.pdf` | `plot_2d_slices` / `plot_vertic_line` | PDF version of the same report (one row per variable, columns: Proxy / THM / Difference for 2-D; one page per variable for 1-D). |
| `displs_stresses_prx.pkl` | `plot_2d_slices` | Cached proxy displacement and stress arrays. Reloaded on the next run when `read_from_cache=True`, skipping proxy recomputation. |

---

## Assumptions & known limitations

### THM solver

- **One segment per perforation** (`model.py`, `reservoir.py`).
  Wells are multi-segment: each perforated cell gets a segment at its own depth, and consecutive segments carry the hydrostatic gravity term, so a zero-rate well holds the formation pressure. The perforated interval is taken from the mesh (the reservoir tag, or the reservoir box).

- **Geomechanics decoupled from flow by default** (`main.py`, `decouple_geomech=True`).
  The volumetric-strain transmissibility terms that feed mechanics back into porosity (and therefore pressure) are zeroed out. This is a common approximation for field-scale problems where the mechanical compaction signal is small compared to the fluid driving force.

- **Temperature scale is relative, not absolute** (`cases/base.py`, `MaoDuan2009Shifted`).
  The OBL range spans −50 to +50 K around a baseline of 0. `MaoDuan2009Shifted` maps that baseline to 373.15 K (100 °C) before calling the correlation. If the actual reservoir temperature differs significantly from 100 °C, `t_abs0` in `MaoDuan2009Shifted.__init__` must be adjusted.

- **Single-phase water physics** (`model.py`, `ModelProperties.evaluate`).
  `ModelProperties.evaluate` uses a trivial flash: water is assumed to be always single-phase liquid.

- **Initial temperature is uniform and equal to zero** (`cases/base.py`).
  `idata.initial.temperature_at_ref_depth = 0` and `idata.initial.temperature_gradient = 0`, so there is no geothermal gradient. The temperature field starts flat and evolves only through injection/production. Enabling a gradient requires setting both parameters and adjusting `t_abs0` in `MaoDuan2009Shifted` so the correlation receives a physically meaningful absolute temperature at every depth.

### Geomechanical proxy

- **Rectangular prism cell shapes** — the proxy integrates the nucleus-of-strain solution over axis-aligned rectangular prisms. Each source cell must be representable as a cuboid with faces perpendicular to X, Y, Z axes, so `main_nucleus_of_strain.py` accepts only hexahedral meshes (`rect_*` cases) and raises `ValueError` for wedge, tetra or pyramid cells (the faulted cases).

- **Mesh aligned with coordinate axes** — the proxy assumes the mesh is not rotated; tilted or curvilinear grids are not supported.

- **Homogeneous rock geomechanical properties** — a single set of elastic constants (Young's modulus, Poisson's ratio, Biot coefficient, thermal expansion coefficient) is used for the entire domain. Spatially varying elastic properties are not accounted for in the proxy.

- **2-D proxy comparison is evaluated at Y = 0 only** (`main_nucleus_of_strain.py`, `plot_2d_slices` mode).
  The XZ slice through the centre of the domain is the only plane compared. Off-centre or XY slices are not plotted.

- **Proxy source cells**: reservoir-only vs. full domain (`main_nucleus_of_strain.py`, `perm_non_rsv` / `poro_non_rsv`).
  When `perm_non_rsv ≤ 0` (truly impermeable overburden) the proxy integrates only over cells with `poro > poro_non_rsv`. When the surrounding rock has any permeability, pressure diffuses into it and all cells are used as sources. The threshold is currently hardcoded to `0`.
