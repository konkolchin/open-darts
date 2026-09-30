---
name: changelog-open-darts
description: Update the open-DARTS CHANGELOG.md after significant modifications with a brief, user-focused entry that links the MR and/or commit, record API changes in the Breaking changes section, and condense existing release blocks. Use after user-visible behavior, model, physics, solver, output, build/CI, or API changes, before handing off or opening an MR, and when asked to revise or shorten a changelog section.
---

# Changelog

Use this skill to keep `CHANGELOG.md` current after significant modifications.
The changelog is written for users of open-DARTS: a short index of what changes for their models,
with links into merge requests and commits where the details live.

## Steps

1. Decide whether the change is significant enough (see the reference). Skip refactors, typo fixes, tests-only and comment-only changes.
2. Edit only the repository-root `CHANGELOG.md`. `darts/CHANGELOG.md` is a gitignored copy made by the build scripts; never edit it.
3. Write from the user's point of view: what is added, fixed, or faster, how it affects their model, and which cases are affected. Leave implementation details to the MR or commit.
4. Add a brief entry (1-2 lines) under the top `# #.#.# [Future]` heading, in the matching area group, and link the MR and/or the commit.
5. If the change touches the public API (renamed, removed, or re-signatured classes, methods, arguments, defaults, CLI flags, file formats, or result-changing defaults), also add it to the `- Breaking changes:` group of the same block, using the Before/Now layout.
6. When asked to revise or shorten an existing block, follow the revision procedure in the reference: classify, draft outside the repo, ask about doubtful items, then apply.
7. Preserve the existing line endings of `CHANGELOG.md`; do not keep line-ending-only rewrites.

## Reference

Read `references/changelog.md` for the entry format, what to keep, shorten, or drop, the Breaking changes layout (copied from the 1.4.0 block), link formats, the revision procedure, and worked examples.
