# New Work Area

This folder is reserved for your original implementation.

Suggested use:

- put new agents here, such as DQN-based variants
- keep new training and evaluation utilities here
- add wrappers around `legacy/` code only when needed
- avoid editing inherited code unless a change is required for integration

A reasonable future structure would be:

```text
src/
  agents/
  envs/
  experiments/
  models/
  utils/
```

This separation makes it easier to explain which parts are inherited and which
parts are your own contribution.
