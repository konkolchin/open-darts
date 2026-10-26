## Code comments vs commit messages
- Keep history and evidence OUT of code comments: when a bug appeared, which commit
  introduced it, what the old code did, measurements, timings, test results,
  before/after numbers. That belongs in the commit message.
- Code comments explain the code as it stands now: what it does, why it is this way,
  what would break if it were written the obvious way. Write them so they stay true
  after the surrounding history is forgotten.
- Example - not in a comment: "the test used to be len(x) == 1, which is never true,
  so a plain box silently got NX//2 cells". In a comment: "three intervals only when
  z_minus_hybrid inserted the midpoints; a plain box has one interval and takes all NX".
  The first sentence goes in the commit message.
