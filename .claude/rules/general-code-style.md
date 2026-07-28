---
paths:
  - "miles/**/*.py"
  - "scripts/**/*.py"
  - "tools/**/*.py"
  - "train.py"
  - "train_async.py"
---

# General Code Style

Default conventions for new and substantially modified Python code in Miles core
modules and scripts. Prefer these unless there is a concrete reason not to.
Preserve established framework contracts and avoid unrelated refactors; call out
deviations in review.

- **Prefer stateless.** Favor pure functions over methods that mutate instance
  state. Keep state in lifecycle objects that are intentionally responsible for
  it, such as rollout, training, tracking, and distributed-runtime managers.
- **Prefer immutable.** Default to immutable data and read-only inputs. Mutate
  only when there is a clear need and the owner of the state is explicit.
- **Initialize stable derived values once.** When a derived value's inputs are
  stable for an object's lifetime, compute it at the earliest lifecycle point
  where its dependencies are valid: in `__init__` for construction-time
  configuration, or in a single explicit setup or lazy-initialization point for
  distributed resources. Store it under a meaningful name. If the inputs can
  change, recompute it where needed or funnel updates through one clear override
  point.
- **Functions stay small.** Keep each function under roughly 100 lines; split
  larger functions into cohesive, well-named helpers.
- **Files stay small.** Keep each file under roughly 1,000 lines; split larger
  modules along cohesive boundaries.
- **Core functions read like pseudocode.** Keep the main orchestration function
  of a unit short and make its algorithm obvious. Push details into well-named
  helpers.
- **Prefer composition without fighting established abstractions.** For new
  behavior, prefer explicit composition or plain functions. Use inheritance or a
  mixin when required by a framework contract, consistent with an established
  Miles architecture, or when it represents cohesive behavior shared by
  multiple implementations.
- **Keep public APIs intentional.** Default implementation details to protected
  names, but preserve established public, plugin, and framework interfaces. Do
  not rename public symbols without a migration plan.
- **Prefer keyword arguments where they improve clarity.** Use keywords for
  internal calls with multiple or ambiguous arguments, especially configuration
  values and boolean flags. Preserve positional calling when it is conventional
  for a mathematical or tensor API, required by an external framework or
  override signature, or clearly more readable.
- **Pass what the callee needs.** Prefer specific values over an entire large
  object. Pass a configuration or context object when that object is the
  established contract or carries many tightly coupled values. Treat it as
  read-only unless mutation is explicitly part of the contract.
- **Keep imports at the top.** Prefer module-level imports at the beginning of
  modules and scripts. Use a local import only when required, such as to break a
  dependency cycle, defer an optional dependency, or avoid a demonstrated
  startup cost, and make the reason clear.
- **Use absolute imports.** Prefer absolute project imports over relative imports
  so dependencies remain explicit and code is easier to move and search.
