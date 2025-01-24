<p align="center" width="100%"><img width="256px" src="logo.png"></p>

# Overview

MOMA is a mobility mapping toolbox. When provided with GIS and simulation data
from multiphysics simulation, it allows for the generation of mobility maps and
operational path planning to optimise traversal speed under the specified
conditions.

# Getting started

From the root of the repository, install and activate the Poetry environment for
the current session:

```shell
poetry install --with documentation
poetry shell
```

Change the directory to the `docs`folder and run the Sphinx Makefile with `cd
docs && make html`.

Once the build process is complete, the documentation will be available under
`docs/build/html/index.html`.