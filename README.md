# Overview


Pygadget is a python module for handling GADGET3 binary snapshots. Its goal is
to provide a fast and intuitive way to access cosmological simulation data.

## Requirements

There is only one requirement:

* [Numpy][numpy]

## Basic usage

Snapshots can be easily access using the `Simulation` class:
    
    from pygadget import Simulation
    snap = Simulation("filename")

Optionally one can signal the presence of blocks set in the makefile by
passing one or more keyword arguments during object initialization. For
example::

    snap = Simulation("filename", pot=False, accel=False, endt=False, tstp=True)

After initialization the snapshot header information can be access as object
attributes. For example:

    snap.h

    snap.omega_matter

    snap.omega_lambda

    snap.particle_numbers

For convenience a summary of the snapshot properties can be display by
printing the ``Simulation`` class instance:

    print(sim)

## Reading blocks

For performance reasons blocks are only read on demand, for a specified
particle type. The ``read_block()`` method is used for
this task:

    gas_pos = snap.read_block("pos", "gas")

Accepted keywords for block types are:

* "pos"
* "vel"
* "id"
* "mass"
* "u"
* "rho"
* "ne"
* "nh"
* "hsml"
* "sfr"
* "age"
* "metals"
* "pot"
* "accel"
* "endt"
* "tstp"
* "esn"
* "esncold"

And accepted keywords for particle types are:

* "gas"
* "halo"
* "disk"
* "buldge"
* "stars"
* "bndry"

Most block types return a 1D ndarray. The exceptions being "pos", "vel" and
"accel" which return a 2D ndarray with three columns representing Cartesian
axis. And "metals" which return a 2D array with 12 columns representing each of
the chemical elements consider in the simulation.

Slicing and array operations can be used to manipulate data. For example:

    composition = snap.read_block("metals", "stars")
    composition[:,0]

Should return the helium mass content for every stellar particle in the
snapshot. Similarly:

    halo_pos = snap.read_block("pos", "halo")
    pos[:,1]

Should return the 'y' coordinate for every halo particle.

## Caveats

* Very early version!!!
* No multi-file support yet
* Endianness was considered but is untested
* I was using python3 during development and didn't pay too much attention to backwards compatibility.

[numpy]: http://www.numpy.org/