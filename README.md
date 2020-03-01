# point-registration-with-relaxation
This repo implements a convex relaxation of the binary optimization problem discussed in this [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.910&rep=rep1&type=pdf), Section 5.4.  The repo's main contribution is to provide the core optimization engine for [NMSAC](https://github.com/jwdinius/nmsac), which seeks to employ the sample consensus strategy of [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus), but with higher quality transformation hypotheses than random sampling.

This repo also implements helper utilities for aligning point clouds after point correspondences have been identified.

## Brief Intro
### Correspondence Identification (Matching)
This work addresses the issue of point cloud matching using a convex relaxation of a \[0, 1\] optimization problem.  There is one _important_ difference between the work presented in the [original paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.910&rep=rep1&type=pdf) and the work presented herein:

> The reference formulation restricts $$m$$, the number of source points, to be less than the number of target points, $$n$$ AND that _all_ $$m$$ source points must be matched with a target point.  The present work allows for $$m \le n$$ AND the ability to match $$k \le m$$ source points to _unique_ target points.

After solution of the fully relaxed optimization problem, the resulting optimum may not be a valid solution of the original \[0, 1\] problem, and so the linear programming solver from Google's [ORTools](https://developers.google.com/optimization) library is used to project that optimum onto the space of valid solutions.

### Transformation Utilities
In addition to computing the optimal correspondences, the best homogeneous transformation between point clouds is computed using [Kabsch's algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm).  For handling outliers and noise, an implementation of the [Iterative Closest Point](https://en.wikipedia.org/wiki/Iterative_closest_point) algorithm is included which allows the user the flexibility to remove a configurable amount of outliers.

### About the Repo
The code is well-commented with unit tests provided for key functionality but some things may remain unclear.  If this is the case, please to make an issue.

#### Static code checking
For desired formatting, please see the script [linter.sh](scripts/linter.sh) and the [`cpplint` docs](https://github.com/cpplint/cpplint).

I will eventually add Travis integration to check each PR, but until then I use the linter script.

## Quick Start

The recommended approach is to use [`docker`](https://docs.docker.com/install/linux/docker-ce/ubuntu/), however I realize that not everyone is familiar with it.  For those users, check out the `RUN` steps in this [file](docker/deps/Dockerfile.std) to properly configure your workspace, including all steps needed to build dependencies from source.

### With `docker` (recommended)
#### Build images
You will first need to build the `nmsac-deps` container.  To do this, execute the following

```shell
$ cd {repo-root-dir}/docker/deps
$ #ln -s Dockerfile.std Dockerfile  ## NOTE: do this if you do not have nvidia-docker2 installed
$ ln -s Dockerfile.nvidia Dockerfile ## NOTE: do this if you do have nvidia-docker2 installed AND you want to use the nvidia runtime
$ ./build-docker.sh  ## this will take awhile to build
```

Now that you have `nmsac-deps` built locally, you can build the `qap-register` image.  This image is incremental, and basically just sets up a user environment for working with this repo.  To build the `qap-register` image, execute the following:

```shell
cd {repo-root-dir}/docker
$ ./build-docker.sh
```

#### Launch development container

```shell
cd {repo-root-dir}
$ ./docker/run-docker.sh {--runtime=nvidia}  ## only add the optional command line arg if you have the nvidia runtime available
```

You should now have an interactive shell to work from.

#### Build the library

```shell
$ cd {repo-root-dir}
$ rm -rf build && mkdir build && cd build && cmake .. && make
```
#### Test

To run the unit tests:

```shell
$ cd {repo-root-dir}/build  ## after following build steps above
$ make test
```

To see graphical output (from python 2.7):

```shell
$ cd {repo-root-dir}/build  ## after following build steps above
$ export PYTHONPATH=$(pwd):$PYTHONPATH
$ cd ../scripts
$ python wrapper-test.py ## or wrapper-test-mincorr.py
```

Inside of these scripts, if the `run_optimization` and `make_plots` flags are set to `True` (and you have the right environment setup for GUI applications), you will see some plots.

## Sample Output (no noise)

### Plot 1:  Optimal Solution
![](./figures/solution-nonoise.png)

_It's worth noting the quality of the solution is much better than results reported in the original paper; matches are much more prominent._

### Plot 2: Correspondences
![](./figures/correspondences-nonoise.png)

### Plot 3: Transform Source Points onto Target Set
![](./figures/transformation-nonoise.png)

_Note: The rate of correct correspondence matching is 100% for the example tested, hence the perfect overlap of the transformed source point set onto the target point set._
