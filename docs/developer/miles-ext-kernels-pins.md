---
title: miles-ext-kernels Op Pins
description: Serve one miles-ext-kernels op from its own submodule commit while every other op stays on 3rdparty/miles-ext-kernels.
---
The miles-ext-kernels submodule (`3rdparty/miles-ext-kernels`, Python package `miles_ext_kernels`) holds the kernels
behind `--use-miles-ext-kernels`, in the namespace `miles_ext_kernels.cake`:

| Path in `miles_ext_kernels/cake/` | What it is | Owner |
| --- | --- | --- |
| `<op>.py` | the op's module: its declaration (`OP`) and entry points | the op |
| `_impl/<op>/` | the op's implementation | the op |
| `_vendor/<pkg>/` | a vendored kernel package; `_vendor/VENDORED.json` lists each one under its op | the op that lists it |
| `_runtime/` | arguments, variants, the kernel loader, prebuild, the workspace pool, release | shared by every op |

A submodule bump moves every op at once; a pin moves one.

## Pinning one op to another miles-ext-kernels commit

A pin is a second checkout of the miles-ext-kernels repository at `3rdparty/miles-ext-kernels-pins/<op>`, named after
the op it serves (`gdn`, `kda`, `conv1d`, ...: the names of the `<op>.py` modules). Use one to:

- keep an op at a known-good commit while the submodule moves on, for example after a regression in that op;
- take a newer commit of one op without moving the others.

Each op can have its own pin, at its own commit. Without a pin nothing changes. Pins are ordinary git submodules:

```bash
# add: serve kda from <commit>
git submodule add https://github.com/radixark/miles-ext-kernels.git 3rdparty/miles-ext-kernels-pins/kda
git -C 3rdparty/miles-ext-kernels-pins/kda fetch origin <commit>
git -C 3rdparty/miles-ext-kernels-pins/kda checkout <commit>
git add .gitmodules 3rdparty/miles-ext-kernels-pins/kda

# move: serve kda from <other-commit>
git -C 3rdparty/miles-ext-kernels-pins/kda fetch origin <other-commit>
git -C 3rdparty/miles-ext-kernels-pins/kda checkout <other-commit>
git add 3rdparty/miles-ext-kernels-pins/kda

# remove: kda follows 3rdparty/miles-ext-kernels again
git rm 3rdparty/miles-ext-kernels-pins/kda
```

`git submodule add` clones only the branches and tags of the repository, so fetch the commit by its full SHA before
checking it out; a commit of a pull request or a fork is fetched the same way. Other clones and CI fetch it by SHA in
`git submodule update`. `git rm` keeps the submodule's git directory under `.git/modules/`, so to pin the same op
again in that clone, add it with `git submodule add --force`.

Commit each change as you would a submodule bump. Other clones pick a pin up with `git submodule update --init`.
`.gitmodules` is the source of truth: only the submodules it registers under `3rdparty/miles-ext-kernels-pins` are
pins. Git leaves the directory of a removed submodule behind in other clones (and in the workspaces of CI runners);
miles ignores such a directory and warns with the command that deletes it.

## How a pinned op loads

`import miles` installs an import hook (`miles/utils/ext_kernel_pins.py`) at the front of `sys.meta_path`. For a
pinned op it loads the modules the op owns from `3rdparty/miles-ext-kernels-pins/<op>/miles_ext_kernels/cake/`:

- `miles_ext_kernels.cake.<op>`;
- `miles_ext_kernels.cake._impl.<op>` and its submodules;
- `miles_ext_kernels.cake._vendor.<pkg>` and its submodules, for every package the pin's own `VENDORED.json` lists
  under the op, so a commit that adds or renames a package of the op works too. Vendored code that imports a sibling
  package as `..cake.<pkg>` reaches the same module, through the runtime's alias.

The top-level package, `miles_ext_kernels.cake`, the runtime and the other ops load from `3rdparty/miles-ext-kernels`.
The runtime registers each op's arguments from the op's module, so a pinned op brings its options and variants with
it, and `configure`, `prebuild` and `release` reach the pinned implementation.

The hook applies in every process that imports miles before an op of `miles_ext_kernels`: the driver, the Ray workers
and pytest, whose `tests/conftest.py` imports miles. Importing `miles_ext_kernels` or its runtime first is fine. A
process that imports a module of a pinned op first loads it from the base, and its later `import miles` fails with an
error that names the module. A script that imports an op directly, such as a test run with `python3 <file>`,
therefore puts `import miles` above that import.

Importing a module of `miles_ext_kernels.cake` fails with a clear error when:

- a pin is not named after an op of the base checkout (a module `miles_ext_kernels/cake/<op>.py`);
- a pin has no `miles_ext_kernels/cake/<op>.py`, for example a submodule that is registered but not checked out, or
  a commit from before the `miles_ext_kernels.cake` layout;
- a pin's `VENDORED.json` cannot be read, or two pins list the same vendored package;
- `MILES_EXT_KERNELS_PINS_DIR` is set but is not the absolute path of a directory.

## Shared modules

A pinned op imports the runtime absolutely (`from miles_ext_kernels.cake._runtime import loader`), so it runs against
the base's runtime, and the pin's own copy of the runtime is never imported. Pin a commit whose op works with the
base's runtime. When a pinned op loads, miles warns if the pin's copy of the shared files differs from the base's,
and names the files. The shared files are those of the base's `miles_ext_kernels/` other than the modules the ops own,
`VENDORED.json` and the list of ops in `_runtime/registry.py` (`OPS`), which every op's commit extends. A pinned op
that calls a runtime function the base has changed then fails on import or while registering its arguments, right
after that warning.

## Kernel builds

The runtime's loader compiles every vendored library with nvcc and caches it in
`${TVM_FFI_CACHE_DIR:-~/.cache/tvm-ffi}/miles_ext_kernels/<package>/` under a name that carries the digest of its
source closure and a hash of the compile flags, nvcc, tvm-ffi and the machine. A pinned and a base version of an op
share a library only when all of these match, so a pin needs no cache of its own. Triton keys its cache
(`TRITON_CACHE_DIR`) by the kernel source in the same way.

## CI

CI checks the repository out with its submodules, pins included. Nothing goes on `PYTHONPATH` for a pin: the hook
finds it next to the miles source tree. A test that imports a pinned op after miles runs the pin, and a test script
that imports a pinned op before miles fails, as described above. A test that runs inside one miles-ext-kernels
checkout, such as the submodule's own test suite, tests that checkout as a whole.

## Which commit an op came from

- Every process that loads a pinned op logs one line with the pin and its commit, read from the checkout's git
  metadata: `miles-ext-kernels: kda loads from its pin <path>/3rdparty/miles-ext-kernels-pins/kda at <commit>`. In
  the driver it comes when the runtime registers the op's arguments.
- A `--use-miles-ext-kernels` run also logs one line at startup, after checking that each pinned op loaded from its
  pin: `miles-ext-kernels: base <path>/3rdparty/miles-ext-kernels at <commit>; kda pinned to
  <path>/3rdparty/miles-ext-kernels-pins/kda at <commit>`.
- `git submodule status` lists the commit each submodule records; a leading `+` marks a checkout at another commit.
- `python -c "import miles, miles_ext_kernels.cake.kda as m; print(m.__file__)"` prints where an op loads from.

## Outside the source tree

The hook reads `.gitmodules` and `3rdparty/miles-ext-kernels-pins` next to the `miles` package it is part of. When
miles runs from another location, set `MILES_EXT_KERNELS_PINS_DIR` to the pins directory in every process that
imports miles: the driver's environment, and the training workers' through `--train-env-vars`. Every checkout in that
directory is a pin. The value must be an absolute path that every node reaches; a relative path, or one missing on a
node, fails the import there.
