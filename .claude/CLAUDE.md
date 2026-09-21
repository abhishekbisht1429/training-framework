# Getting session context
## STATE.md
 - Read .context/STATE.md to get the context. Do not read the project files unless explicitly asked to do so.
 - If you have to read the project files to get something done. Ask for permission.
 - After making changes, update STATE.md so that next session can pickup from there if needed.
 - Keep STATE.md as concise as possible. You can do this by removing past state information actively from it and only keeping info that is relevant currently in it.

## HANDOFF.md
 - Use .context/HANDOFF.md to save some context for unfinished tasks. Keep this file as concise as possible.

## Making Local changes
- Whenever any change in the codebase is requested, even if a small change, do not implement it immediately unless explicitly asked to do so. First create a plan as ask to review it before implementing it.
- Unless explicitly told not to do so, you can make changes in the current checkout branch. You don't have to create some background worktree for that.
- Never commit changes by yourself. Always ask permission to do so.
- Individual test cases should not be modified unless absolutely necessary. When you have to modify an existing test, always ask for permission.

### Writing new tests
- Make sure that the test cases use only the public api wherever possible. Do not use internal members if there is an alternative.

## Running commands
- Unless required, run all commands in silent mode without any verbose, wherever possible.

## Learning
- When you learn something generic from your sessions, add it to .context/LEARNING.md.
