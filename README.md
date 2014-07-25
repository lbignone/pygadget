# Overview


Pygadget is a python module for handling GADGET3 binary snapshots. Its goal is
to provide a fast and intuitive way to access cosmological simulation data.

## Requirements

* [Numpy][numpy]
* [Pandas][pandas]

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

Most block types return a single column Pandas Dataframe. The exceptions being
"pos", "vel" and "accel" which return a dataframe with three columns ('x',
'y', 'z') representing Cartesian axis. And "metals" which return a DataFrame
with 12 columns representing each of the chemical elements consider in the
simulation.

Check the Pandas [Intro to Data Structures](http://pandas.pydata.org/pandas-docs/dev/dsintro.html) for a primer on DataFrames manipulation and slicing.

For example 

    composition = snap.read_block("metals", "stars")
    composition.He

Should return the helium mass content for every stellar particle in the
snapshot. Similarly:

    halo_pos = snap.read_block("pos", "halo")
    pos.y

Should return the 'y' coordinate for every halo particle.

As blocks are index by particle ID, you should be careful on how you select
rows. More information on the [Idexing/Selection](http://pandas.pydata.org/pandas-docs/dev/dsintro.html#indexing-selection) section of the Pandas documentation.


## Subfind

The Subfind interface works very similarly to snapshots. The following code
sets the subfind objects

    sub = Subfind(basedir=dir, num=0, snap=snapshot)

where `dir` is the base directory for subfind folders, `num` the desired
subfind folder and `snapshot` the associated Simulation object snapshot.

All subfind catalog properties can be access trough object attributes like

    sub.nshubalos

    sub.sublen

Subhalo particles can be read with

    pos = sub.read_block_by_subhalo(block_type, particle_type, subhalo_number)

The output is the same as Simulation.read_block, but particles are filtered
down to the ones in the required subhalo

A few extra methods are included

    sub.optical_radius(subhalo, factor=0.83, rcut=30.0)

to calculate subhalos optical radius and

    sub.mass_inside_radius(self, radius, subhalo, particle_keys=particle_keys)

to calculate the mass inside a give radius


## Caveats

* No multi-file support yet
* Endianness was considered but is untested

[numpy]: http://www.numpy.org/
[pandas]: http://pandas.pydata.org/