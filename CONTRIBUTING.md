# Contributing

Small, testable changes are easiest to review. Please keep the command-line
interface backwards compatible when possible and update the argument table in
`README.md` when adding an option.

Before opening a pull request:

```sh
make
make test
python3 -m py_compile visualizer.py profiles.py gui.py make_preview.py point_sheet.py
```

Changes to the native renderer should include a focused correctness test and,
when performance is the goal, a before/after `make benchmark` result. Do not
commit songs, rendered videos, keyframe caches, or local build products; the
repository ignore rules cover the usual extensions.

The project is licensed under the MIT License.
