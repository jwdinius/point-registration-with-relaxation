# point-registration-with-relaxation
binary optimization relaxation applied to the point set registration problem

## Brief Intro
This repo addresses the issue of point cloud matching using a convex relaxation of a \[0, 1\] optimization problem.  For more information about the formulation of the problem, see this [reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.910&rep=rep1&type=pdf); in particular Section 5.4.

In addition to computing the optimal correspondences, the best homogeneous transformation between point clouds is computed using [Kabsch's algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm).

The code is well-commented, but some things may remain unclear.  If this is the case, feel free to make an issue.

## Quick Start
_NOTE: Graphical applications with docker are only supported for the Nvidia image/launch-script combination.  Comparable modifications can be made to [Dockerfile.apponly](./docker/Dockerfile.apponly) to make it work.  If you add this capability, submit a PR and I will merge it._

The quickest way to get started is by building a docker image using one of the provided [Dockerfiles](./docker).  All of the dependencies will be installed, and so, in my opinion, this is definitely the preferred way to get started.  If you don't want to use docker, you can manually perform all of the steps listed in the Dockerfile to install dependencies.

After building the image, modify one the provided [launch scripts](./scripts) to use the image you just built and run it.  This will open an interactive session with a container based on your image.  _Note: as with the Dockerfile: the script you use will depend on whether or not you have the nvidia-runtime available._

Now that you are in the container, you can build the package:

```shell
cd {path-above-registration}/registration
mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make test  # optional: run this if you wish to run C++ unit tests
```

If you want to quickly modify parameters, including the sizes of the source and target point sets, the main routine is wrapped in a python module; see [this script](./scripts/wrapper-test.py).  To run:

```shell
# execute after building
sudo ldconfig  # only need to run this once (per container launch, if using docker)
export PYTHONPATH=$PYTHONPATH:{path-to-registration}/registration/build
cd {path-above-registration}/registration/scripts
python wrapper-test.py
```

If the `run_optimization` and `make_plots` flags are set to `True` (and you have the right environment-setup for GUI applications, you will see some plots.

_For explanations about the variable conventions, see the paper referenced above._

## Plot 1:  Optimal Solution
![](./figures/solution-nonoise.png)

_It's worth noting the quality of the solution is much better than results reported in the original paper; matches are much more prominent._

## Plot 2: Correspondences
![](./figures/correspondences-nonoise.png)

## Plot 3: Transform Source Points onto Target Set
![](./figures/transformation-nonoise.png)

_Note: The rate of correct correspondence matching is 100% for the example tested, hence the perfect overlap of the transformed source point set onto the target point set._
