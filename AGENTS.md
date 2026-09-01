# Codex project instructions

## Hand-off files
- Hand off files are at `.codex/handoff`
- Check the files here to get context.
- When writing new handoff file always include a time stamp in it.

## Intermediate results instructions
- These instructions apply even for subagents or any other parallel agents.
- If the detailed output log of any command is not required then always suppress it.

## New test cases instructions
- Do not add very implementation speciifc test cases. 
- Wherever possible add future agnostic tests that test the intended functionality rather than testing whether the component behaves in a implementation specific way.
- Every builtin component must have some serialization and deserialization tests.

## Parallel test guardian

- After every completed major change, the primary agent must spawn the project-scoped
  `test_guardian` agent as a parallel subagent. A major change includes a feature,
  behavior-affecting bug fix, meaningful refactor, dependency or configuration change,
  or a change to the registry, component lifecycle, session execution, or persistence.
- After every commit created during an active Codex workflow, immediately spawn
  `test_guardian` against the committed `HEAD`, even if tests ran before the commit.
- Give the subagent the completed change's scope, relevant files, and whether it is
  validating a working-tree checkpoint or a commit. Do not start a duplicate guardian
  for the same revision.
- The primary agent may continue independent, non-overlapping work while the guardian
  runs, but must wait for its result before making another commit, modifying overlapping
  files, or reporting the task complete.
- The guardian owns test execution and failure diagnosis. It must not modify production
  code. It may modify tests only under the strict necessity rules in its custom-agent
  instructions; otherwise it reports the production defect to the primary agent.
- A failed guardian run blocks completion unless the failure is clearly pre-existing or
  environmental and is reported with evidence. Test changes made by the guardian must
  remain visible for primary-agent review and must never be silently folded into an
  existing commit.
- Use the IDE-configured Python interpreter as required by the `python-tools` skill.
  Prefer focused tests while iterating; after a major change run the full suite, and
  after every commit always run the full suite.

This policy applies to Codex workflows while they are active. Commits made externally
while Codex is not running cannot start a subagent; CI remains the repository-wide
fallback for those commits.
