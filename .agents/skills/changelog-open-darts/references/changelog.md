# Skill: Changelog

## Overview

`CHANGELOG.md` in the repository root is the release-facing list of what changed.
It is written for users of open-DARTS, not for its developers: it says what a user gains or
has to change in their models, and points to the MR and/or commit for the details.

Do not edit `darts/CHANGELOG.md`: it is listed in `.gitignore` and is overwritten by
`helper_scripts/install_darts.*` and `build_darts_cmake.*` from the root file.

---

## When to add an entry

Add an entry after a modification that a user, model author, or reviewer would want to know about:

- new feature, model, physics, solver option, output or tool
- bug fix that changes results or behavior of existing models
- performance improvement that users will notice
- changed defaults, numerical behavior, or reference-data baselines
- API changes (see Breaking changes below)
- build, packaging, supported-Python, or CI/CD changes

Skip:

- refactors with no visible effect, renames of private helpers
- tests-only, comment-only, formatting-only, and typo changes
- edits to an unmerged feature that already has an entry in the `[Future]` block (update that entry instead)

Search `CHANGELOG.md` for the MR number or a keyword first to avoid duplicates.

---

## Where it goes

The top of the file holds the open block:

```text
# #.#.# [Future]
```

Put new entries there, never under a released version. Do not change version numbers or dates;
the release process does that. If the `[Future]` heading is missing, add it as the first line.

Group entries by area, as in the released blocks (for example `1.4.0`). Reuse an existing
group in the `[Future]` block if there is one; otherwise add it:

- `- OBL and operators:`
- `- Physics:`
- `- Reservoirs:`
- `- Output:`
- `- Solvers:`
- `- Models:`
- `- Tools:`
- `- Build system and CI/CD:`
- `- Breaking changes:` (always last)

Sub-bullets are indented two spaces under the group.

---

## Write for the user

Focus on the user's experience, not on the implementation. For every entry answer:

- what is added, fixed, or faster
- how it affects the user's model (what they must change, what results move, what to re-baseline)
- which cases are affected (physics type, model names, reservoir type, platform, resolution, solver)

Keep:

- new features, and how to switch them on (say when they are off by default)
- fixes that change results, with the affected cases
- performance gains, with the case and a number only when it helps the reader
- changed defaults and behavior, with the action needed (for example "re-baseline reference data")
- API changes with a migration path, in the Breaking changes group
- new build, install, or dependency requirements

Drop:

- implementation details: class and template structure, C++ internals, algorithm derivations, internal renames
- tests and test coverage
- CI job internals, unless the workflow of users or developers changes
- diagnostics and internals with no user-visible effect
- fixes for bugs that were introduced and fixed inside the same unreleased block
- statements superseded later in the same block (for example an earlier dependency bump)

Shorten:

- one or two lines per entry; move analysis, rationale, and measurements to the MR or commit
- replace long code blocks with a Before/Now pair, or point to a migrated model in the repo
  (for example ``see `set_iapws_physics()` in `models/CoaxWell/model.py` ``)
- use current names; if something was renamed later in the same block, describe only the final state
- do not repeat the same change in a feature entry, a breaking-change entry, and a migration guide;
  give the feature once and the migration once, in the Breaking changes group

When in doubt whether to keep, shorten, or drop something, or where migration material should live,
ask the user instead of guessing.

Rewriting an implementation-focused bullet for users:

```text
Before: `discretizer.cpp` and `linalg/matrix.h` call `abs()` on `double`s without including `<cstdlib>`,
        so `abs(double)` bound to `int abs(int)` ... (several more lines of mechanism and history)
After:  Fixed MPFA discretizer distances and SVD pivots being truncated to integers. Coarse grids were
        unaffected; at ~32k cells FS-CPR goes from exhausting a 5000-iteration budget to ~14 iterations
        per Newton, and MPFA results at fine resolution change.
```

---

## Entry format

- One or two lines. State what changed and, when it matters, the effect. No timings or long
  rationale; that belongs in the MR or commit message.
- End with a link. Use the MR when there is one, the commit otherwise, or both:

```text
([!NNN](https://gitlab.com/open-darts/open-darts/-/merge_requests/NNN))
([`abc1234`](https://gitlab.com/open-darts/open-darts/-/commit/abc1234))
```

- Never invent an MR number or commit hash. If the MR does not exist yet, use the commit
  after committing, or ask the user for the MR number.

---

## Breaking changes

An API change goes into the `- Breaking changes:` group of the same `[Future]` block, in
addition to its normal entry. Count as API changes: renamed or removed public modules,
classes, methods, functions, arguments or attributes; changed signatures, argument order,
or defaults; changed CLI flags or file formats; and changes that alter results of existing
models.

Keep all breaking changes of a release in this one group, not in separate sections per MR,
with the MR link on each bullet. Use the layout of the `1.4.0` block: a bullet stating the
change, then one `Before` and one `Now` line in GitLab diff syntax. Each line ends with a
backslash so the Markdown keeps the line break, and the last `Now` line does not.

```text
- Breaking changes:
  - Rename an input argument of the method `add_well`:\
  {- Before: self.reservoir.add_well(..., wellbore_diameter) -}\
  {+ Now:    self.reservoir.add_well(..., well_diameter) +}
  - Plotting methods for structured data `plot_xarray()` has become `output_to_plt()` and has options to plot from solution file, xarray dataset or engine.X:\
  {- Before: self.output.plot_xarray(xarray_data, output_properties, timestep, ...) -}\
  {+ Now:    self.output.output_to_plt(sol_filepath, xarray_data, output_properties, timestep, ) +}
```

When several arguments of one API change together, put a lone `\` line between the pairs, as
in the `Compositional(...)` / `PropertyContainer(...)` / `OperatorsBase(...)` example of `1.4.0`.
A `Now` block that needs several lines keeps them without trailing backslashes inside the
`{+ ... +}` block.

For a change with no code form (for example a removed option), state the migration in one
sentence instead of a Before/Now pair. For a larger migration, keep the few notes a user
needs and refer to a migrated model in the repo (or a docs page) instead of a code block.
Also call out compatibility impact as required by `AGENTS.md` (constructor signatures,
parameter ordering, CLI flags, file formats, packaging).

---

## Revising an existing block

Use this when asked to revise, shorten, or clean up a release block.

1. Read the whole block first, including its migration guides and notes.
2. Classify every item as keep, shorten, or drop, using "Write for the user" above.
3. Draft the new block in a separate file outside the repository. Do not edit `CHANGELOG.md` yet.
   Report the size before and after, and list what was dropped.
4. Check facts against the source text and the code: do not add claims that are not in the
   original block, and verify that names and paths you mention (for example a model file) exist.
5. Ask about the doubtful items: whether a released version may be edited in place, what happens
   to long migration examples, and whether the Breaking changes group is consolidated.
6. After the answers, replace the block in the root `CHANGELOG.md` in place, preserving the
   existing line endings.
7. Verify the result: everything outside the block is identical to `HEAD`, the file has no
   mixed line endings, and every MR link of the original block that is still relevant is kept.
8. Leave `darts/CHANGELOG.md` alone and say that this build copy is stale until the next build.

---

## Examples

Model fix:

```text
- Models:
  - fracture_network: rate-controlled wells now use the configured targets and the BHP constraint is set on `w.constraint`. ([!NNN](https://gitlab.com/open-darts/open-darts/-/merge_requests/NNN))
```

Feature:

```text
- Output:
  - Added `Output.output_to_plt()` for structured reservoirs. ([!169](https://gitlab.com/open-darts/open-darts/-/merge_requests/169))
```

API change (entry in its area group plus the Breaking changes group):

```text
- Physics:
  - `PropertyContainer` takes `eps_z` instead of `min_z`. ([`abc1234`](https://gitlab.com/open-darts/open-darts/-/commit/abc1234))
- Breaking changes:
  - Renamed an argument of `PropertyContainer`:\
  {- Before: PropertyContainer(..., min_z=zero) -}\
  {+ Now:    PropertyContainer(..., eps_z=epsilon) +}
```

---

## Checklist before hand-off

- entry is in the root `CHANGELOG.md` under `[Future]`, in the right group, 1-2 lines
- it says what changes for the user and which cases are affected, not how it is implemented
- link points to a real MR and/or commit
- API changes are also listed under `- Breaking changes:` in the Before/Now layout
- `darts/CHANGELOG.md` untouched, line endings of `CHANGELOG.md` unchanged
