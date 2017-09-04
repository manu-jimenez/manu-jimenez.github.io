---
layout: post
title: "Data-science toolbox setup "
date: 2017-09-04
---

In this post, I have written down some really simple instructions for setting up a working open-source data-science toolbox
from scratch with **anaconda**. Instructions are for mac but it should be very similar for Linux.

### 1 - Uninstall previous anaconda (*optional*)

Use `anaconda-clean` to remove previous distribution

```
$ conda install anaconda-clean
$ anaconda-clean --yes
```

Then, remove the path `~/Users/USER/anaconda/bin` from your `PATH` variable.

Finally, remove the `anaconda` folder

```
$ rm -rf ~/anaconda
```

### 2 - Fresh anaconda install

Download the package from [here](https://docs.continuum.io/anaconda/install/)

Install it and add the binaries folder to your `PATH`

```
$ export PATH=~/anaconda/bin/:$PATH
```

### 3 - Install the necessary python and R libraries

Anaconda makes this extremely simple, installing almost every necessary library. First python 

```
$ conda install anaconda
$ conda update conda
$ conda update anaconda
```

And now, R

```
$ conda install -c r r-essentials
```

Tensorflow is missing from the anaconda installation, let's install it too

```
$ conda install tensorflow
```

*aaaand this should be enough!*

### 4 - To Do

- Discuss conda environments
