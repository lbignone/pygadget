from struct import unpack
from numpy import fromstring, fromfile, concatenate, array
import pandas as pd
from functools import wraps
#from astropy.utils.console import ProgressBar


particle_keys = [
    "gas",
    "halo",
    "disk",
    "buldge",
    "stars",
    "bndry",
]

element_keys = [
    "He",
    "C",
    "Mg",
    "O",
    "Fe",
    "Si",
    "H",
    "N",
    "Ne",
    "S",
    "Ca",
    "Zi",
]


def memoize(func):
    cache = {}

    @wraps(func)
    def wrap(*args):
        if args not in cache:
            print("hola")
            cache[args] = func(*args)
        return cache[args]
    return wrap


class Simulation:

    """ Base class for  handling GADGET3 snapshots

    Attributes:
        name (str):    File name of the snapshot.
        flags (dict):   Flags to signal presence of certain special blocks.
        particle_keys (list of str):    Names of particle types.
        block_keys (list of str):   Names of block types.
        element_keys (list of str): Names of chemical elements.
        particle_numbers (dict):    Particle numbers of each type.
        mass_number (dict): Particle masses of each type. If 0 for a type
            which is present, individual particles masses are stored in the
            mass block.
        cosmic_time (float):    Time of output, or expansion factor for
            cosmological simulations.
        redshift (float):   redshift
        omega_matter (float):   Matter density at redshift zero in units
            of critical density.
        omega_lambda (float):   Cosmological energy density at redshift
            zero in units of critical density.
        h (float):  Hubble parameter
        particle_total_numbers (dict):  Total number of particles of each
            type
        particle_total_numbers_hw (dict):   Most significant word of 64-bit
            total particle numbers for simulations with more than 2e32
            particles. Otherwise zero.
        swap (char):    Endianness represented by a single character
            ('<': little-endian, '>': big-endian).
    """

    def __init__(self, name, pot=False,
                 accel=False, endt=False, tstp=False):
        """'Simulation' initialization

        Args:
            name (str): Snapshot filename (full path).

            pot (bool): Flag to signal presence of gravitational potential
                block. Defaults to False.
            accel (bool):   Flag to signal presence of acceleration block.
                Defaults to False.
            endt (bool):    Flag to signal presence of rate of change of
                entropic function block. Defaults to False.
            tstp (bool):    Flag to signal presence of timesteps block.
                Defaults to False.
        """

        self.name = name
        self.flags = {}
        self.flags["pot"] = pot
        self.flags["accel"] = accel
        self.flags["endt"] = endt
        self.flags["tstp"] = tstp

        self.particle_keys = [
            "gas",
            "halo",
            "disk",
            "buldge",
            "stars",
            "bndry",
        ]

        self.block_keys = [
            "header",
            "pos",
            "vel",
            "id",
            "mass",
            "u",
            "rho",
            "ne",
            "nh",
            "hsml",
            "sfr",
            "age",
            "metals",
            "pot",
            "accel",
            "endt",
            "tstp",
            "esn",
            "esncold",
        ]

        self.element_keys = [
            "He",
            "C",
            "Mg",
            "O",
            "Fe",
            "Si",
            "H",
            "N",
            "Ne",
            "S",
            "Ca",
            "Zi",
        ]

        self._read_header()

    def _read_header(self):
        """Read snapshot header information and check for file endianness

        Raises:
            NameError:  If block limits do not not match expected size.
        """

        f = open(self.name, 'rb')

        size_swaped = 65536
        size_checked = 256
        size_used = 196
        size_unused = size_checked - size_used

        data = f.read(4)
        size = unpack('i', data)[0]

        if (size == size_swaped):
            self.swap = '>'
        elif (size == size_checked):
            self.swap = '<'
        else:
            raise NameError("Error reading header in file: %s" % self.name)

        s = self.swap

        self.particle_numbers = {}
        for key in self.particle_keys:
            data = f.read(4)
            self.particle_numbers[key] = unpack(s+'I', data)[0]

        self.particle_mass = {}
        for key in self.particle_keys:
            data = f.read(8)
            self.particle_mass[key] = unpack(s+'d', data)[0]

        data = f.read(8)
        self.cosmic_time = unpack(s+'d', data)[0]

        data = f.read(8)
        self.redshift = unpack(s+'d', data)[0]

        data = f.read(4)
        self.flags["sfr"] = unpack(s+'i', data)[0]

        data = f.read(4)
        self.flags["feedback"] = unpack(s+'i', data)[0]

        self.particle_total_numbers = {}
        for key in self.particle_keys:
            data = f.read(4)
            self.particle_total_numbers[key] = unpack(s+'I', data)[0]

        data = f.read(4)
        self.flags["cooling"] = unpack(s+'i', data)[0]

        data = f.read(4)
        self.file_number = unpack(s+'i', data)[0]

        data = f.read(8)
        self.box_size = unpack(s+'d', data)[0]

        data = f.read(8)
        self.omega_matter = unpack(s+'d', data)[0]

        data = f.read(8)
        self.omega_lambda = unpack(s+'d', data)[0]

        data = f.read(8)
        self.h = unpack(s+'d', data)[0]

        data = f.read(4)
        self.flags["stellar_age"] = unpack(s+'i', data)[0]

        data = f.read(4)
        self.flags["metals"] = unpack(s+'i', data)[0]

        self.particle_total_numbers_hw = {}
        for key in self.particle_keys:
            data = f.read(4)
            self.particle_total_numbers_hw[key] = unpack(s+'i', data)[0]

        data = f.read(4)
        self.flags["entr_ics"] = unpack(s+'i', data)[0]

        f.read(size_unused)
        data = f.read(4)
        size = unpack(s+'i', data)[0]

        if (size != size_checked):
            raise NameError("Error reading header in file: %s" % self.name)

        f.close()

        self._file_structure()

        self._file_check()

    def _file_structure(self):
        """Compute block sizes."""

        total_number = 0
        mass_number = 0
        for key in self.particle_keys:
            number = self.particle_numbers[key]
            total_number += number
            if (number != 0 and self.particle_mass[key] == 0):
                mass_number += number

        data_size = 4
        dims = 3
        metals = len(self.element_keys)

        size = total_number * data_size
        size_space = size * dims
        size_mass = mass_number * data_size
        size_gas = self.particle_numbers["gas"] * data_size
        size_stars = self.particle_numbers["stars"] * data_size
        size_baryons = size_gas + size_stars
        size_metals = size_baryons * metals

        self.block_sizes = {}
        self.block_sizes["header"] = 256
        self.block_sizes["pos"] = size_space
        self.block_sizes["vel"] = size_space
        self.block_sizes["id"] = size
        self.block_sizes["mass"] = size_mass
        self.block_sizes["u"] = size_gas
        self.block_sizes["rho"] = size_gas
        if self.flags["cooling"]:
            self.block_sizes["ne"] = size_gas
            self.block_sizes["nh"] = size_gas
        self.block_sizes["hsml"] = size_gas
        if self.flags["sfr"]:
            self.block_sizes["sfr"] = size_gas
        if self.flags["stellar_age"]:
            self.block_sizes["age"] = size_stars
        if self.flags["metals"]:
            self.block_sizes["metals"] = size_metals
        if self.flags["pot"]:
            self.block_sizes["pot"] = size
        if self.flags["accel"]:
            self.block_sizes["accel"] = size_space
        if self.flags["endt"]:
            self.block_sizes["endt"] = size_gas
        if self.flags["tstp"]:
            self.block_sizes["tstp"] = size
        if self.flags["feedback"]:
            self.block_sizes["esn"] = size_baryons
            self.block_sizes["esncold"] = size_baryons

    def _file_check(self):
        """Check the file structure for potential errors.

        Raises:
            NameError:  If block limits do not not match expected size.
        """

        s = self.swap

        f = open(self.name, 'rb')

        for key in self.block_keys:
            if key in self.block_sizes.keys():
                size = self.block_sizes[key]

                data = f.read(4)
                size_check = unpack(s+'i', data)[0]

                if (size_check != size):
                    raise NameError("Error in block: %s" % key)

                f.seek(size, 1)

                data = f.read(4)
                size_check = unpack(s+'i', data)[0]

                if (size_check != size):
                    raise NameError("Error in block: %s" % key)

        f.close()

    @memoize
    def read_block(self, block_type, particle_type):
        """Read block from snapshot file

        Args:
            block_type (str):   Type of block.
            particle_type (str): Type of particle.

        Returns:
            Numpy ndarray containing the block data for the specified particle
            type.

            The shape of the ndarray is specified by the block type. In the
            case of 'pos', 'vel' and 'accel' blocks, shape = (particle_number,
            3). For the 'metals' block, shape = (particle_number, number of
            elements). All other blocks return a 1-dimensional ndarray.

        Raises:
            NameError:  If block limits do not not match expected size.

        Note:
            This functions only works for data blocks, to access header
            information use the `Simulation` class attributes.
        """

        s = self.swap

        f = open(self.name, 'rb')

        size = self.block_sizes[block_type]

        order = self.block_keys.index(block_type)

        skip = 0
        for key in self.block_keys[0:order]:
            try:
                skip += self.block_sizes[key] + 8
            except:
                pass

        f.seek(skip, 0)

        data = f.read(4)
        size_check = unpack(s+'i', data)[0]
        if (size_check != size):
            raise NameError("Error reading block: %s" % block_type)

        offset, read_size, remainder = self._compute_offset(block_type,
                                                            particle_type)

        f.seek(offset, 1)

        data_block = f.read(read_size)

        f.seek(remainder, 1)

        data = f.read(4)
        size_check = unpack(s+'i', data)[0]
        if (size_check != size):
            raise NameError("Error reading block: %s" % block_type)

        f.close()

        dtype = s+'f4'
        if (block_type in ["id"]):
            dtype = s+'u4'

        block = fromstring(data_block, dtype=dtype)

        ydim = self.particle_numbers[particle_type]
        if (block_type in ["pos", "vel", "acc"]):
            shape = (ydim, 3)
        elif (block_type in ["metals"]):
            dim = len(self.element_keys)
            shape = (ydim, dim)
        else:
            shape = block.shape

        block.shape = shape

        if block_type in ["pos", "vel", "accel"]:
            columns = ['x', 'y', 'z']
        elif block_type in ["metals"]:
            columns = element_keys
        else:
            columns = [block_type]

        if block_type != "id":
            ids = self.read_block("id", particle_type)
            block = pd.DataFrame(block, columns=columns, index=ids.values)
        else:
            block = pd.Series(block, name="id")

        return block

    def _compute_offset(self, block_type, particle_type):
        """Computes offsets from the beginning of the block to the desired
        particles. As well as the size of the section containing the desired
        particles and the size of the remaining portion of the block.

        Args:
            block_type (str):   Type of block.
            particle_type (str): Type of particle.

        Returns:
            (tuple):    (offset, read_size, remainder).
        """

        particle_number = {}

        dim = 1

        if block_type in ["id", "pos", "vel", "accel", "pot", "tstp"]:
            particle_number = self.particle_numbers
            if block_type in ["pos", "vel", "accel"]:
                dim = 3

        elif block_type in ["u", "rho", "ne", "nh", "hsml", "sfr", "endt"]:
            for key in ["gas"]:
                particle_number[key] = self.particle_numbers[key]

        elif block_type in ["esn", "esncold", "metals"]:
            for key in ["gas", "stars"]:
                particle_number[key] = self.particle_numbers[key]
            if block_type in ["metals"]:
                dim = len(self.element_keys)

        elif block_type in ["mass"]:
            for key in self.particle_keys:
                if self.particle_mass[key] == 0:
                    if self.particle_numbers[key] != 0:
                        particle_number[key] = self.particle_numbers[key]

        elif block_type in ["age"]:
            for key in ["stars"]:
                particle_number[key] = self.particle_numbers[key]

        order = self.particle_keys.index(particle_type)
        offset = 0
        for key in self.particle_keys[0:order]:
            if key in particle_number:
                offset += particle_number[key]

        remainder = 0
        for key in self.particle_keys[order+1:]:
            if key in particle_number:
                remainder += particle_number[key]

        dim *= 4
        read_size = particle_number[particle_type] * dim
        offset *= dim
        remainder *= dim

        return offset, read_size, remainder

    def filter_by_ids(self, block_type, particle_type, ids=[]):

        block = self.read_block(block_type, particle_type)

        return block.loc[ids].dropna()

    def __repr__(self):

        first = '<'
        last = '>'
        delimiter = ' | '

        string = first + self.name + delimiter
        for key in self.particle_keys:
            string += key + ': ' + '%d ' % self.particle_numbers[key]
        string += last

        return string

    def __str__(self):

        string = "file: %s\n" % self.name
        string += "file number: %s\n" % self.file_number
        string += "endianess: %s\n" % self.swap
        string += "particle numbers: %s\n" % self.particle_numbers
        string += "particle mass: %s\n" % self.particle_mass
        string += "cosmic time: %s\n" % self.cosmic_time
        string += "redshift: %s\n" % self.redshift
        string += "box size: %s\n" % self.box_size
        string += "omega_matter: %s\n" % self.omega_matter
        string += "omega_lambda: %s\n" % self.omega_lambda
        string += "h: %s\n" % self.h
        string += "flags: %s" % self.flags

        return string


class Fof:

    def __init__(self, basedir, num, snap=None):
        self.basedir = basedir
        self.num = num

        self.name = basedir
        self.name += "/groups_{0:03d}/group_tab_{0:03d}.0".format(self.num)

        self._read_header()
        self._load_catalogue()
        self._load_ids()

        self.snap = snap

    def _read_header(self):

        header_keys = [
            "totngroups",
            "ntask",
        ]
        with open(self.name, 'rb') as f:

            f.seek(8, 0)
            for key in header_keys:
                value = fromfile(f, dtype="i4", count=1)[0]
                setattr(self, key, value)

    def _load_catalogue(self):

        basename = self.basedir
        basename += "/groups_{0:03d}/group_tab_{0:03d}.".format(self.num)

        dims = 3
        n_particle_types = len(particle_keys)

        array_keys = [
            "grouplen",
            "groupoffset",
            "grouplentype",
            "groupmasstype",
            "groupcm",
            "groupsfr",
        ]

        self.ngroups = []
        self.nids = []

        value = {}
        for i in range(self.ntask):
            name = basename + "{0}".format(i)
            with open(name, 'rb') as f:

                ngroups = fromfile(f, dtype="i4", count=1)[0]
                self.ngroups.append(ngroups)

                nids = fromfile(f, dtype="i4", count=1)[0]
                self.nids.append(nids)

                f.seek(2*4, 1)

                data_types = {
                    "grouplen": "({0},)i4".format(ngroups),
                    "groupoffset": "({0},)i4".format(ngroups),
                    "grouplentype": "({0},{1})i4".format(ngroups,
                                                         n_particle_types),
                    "groupmasstype": "({0},{1})f8".format(ngroups,
                                                          n_particle_types),
                    "groupcm": "({0},{1})f4".format(ngroups, dims),
                    "groupsfr": "({0},)f4".format(ngroups),
                }

                for key in array_keys:
                    dt = data_types[key]
                    data = fromfile(f, dtype=dt, count=1)[0]

                    if i == 0:
                        value[key] = data
                    else:
                        value[key] = concatenate([value[key], data])

        for key in array_keys:
            setattr(self, key, value[key])

    def _load_ids(self):
        basename = self.basedir
        basename += "/groups_{0:03d}/group_ids_{0:03d}.".format(self.num)

        self.ids = []
        group = 0
        for i in range(self.ntask):
            name = basename + "{0}".format(i)
            with open(name, 'rb') as f:
                f.seek(4*4, 0)
                for j in range(self.ngroups[i]):
                    ids = {}
                    for p_type_n, particle_type in enumerate(particle_keys):
                        count = self.grouplentype[group, p_type_n]
                        ids[particle_type] = fromfile(f, dtype="u4",
                                                      count=count)
                    self.ids.append(ids)
                    group += 1

    def read_block_by_group(self, block_type, particle_type, group):
        ids = self.ids[group][particle_type]
        block = self.snap.filter_by_ids(block_type, particle_type, ids)

        return block


class Subfind:

    def __init__(self, basedir, num, snap=None):
        self.basedir = basedir + "/postproc_{0:03d}/".format(num)
        self.num = num
        self.snap = snap

        self.basename = basedir
        self.basename += "/postproc_{0:03d}/sub_tab_{0:03d}.".format(self.num)

        folder = "/postproc_{0:03d}/".format(self.num)
        self.locationsname = basedir + folder + "locations.h5"

        self._read_header()
        self._load_catalogue()
        self._load_ids()

    def _read_header(self):

        name = self.basedir + "sub_tab_{0:03d}.{1}".format(self.num, 0)

        self.ngroups = []
        self.nids = []
        header_keys = [
            "totngroups",
            "ntask",
        ]
        self.nsubhalos = []
        with open(name, 'rb') as f:
            f.seek(8, 0)
            for key in header_keys:
                value = fromfile(f, dtype="i4", count=1)[0]
                setattr(self, key, value)

        self.ntask = 1

        for i in range(self.ntask):
            name = self.basename + "{0}".format(i)

            with open(name, 'rb') as f:
                f.seek(0, 0)
                value = fromfile(f, dtype="i4", count=1)[0]
                self.ngroups.append(value)

                value = fromfile(f, dtype="i4", count=1)[0]
                self.nids.append(value)

                f.seek(8, 1)

                value = fromfile(f, dtype="i4", count=1)[0]
                self.nsubhalos.append(value)

    def _load_catalogue(self):

        array_keys = [
            "nsubperhalo",
            "firstsubofhalo",
            "sublen",
            "suboffset",
            "subparenthalo",
            "halo_m_mean200",
            "halo_r_mean200",
            "halo_m_crit200",
            "halo_r_crit200",
            "halo_m_tophat200",
            "halo_r_tophat200",
            "subpos",
            "subvel",
            "subveldisp",
            "subvmax",
            "subspin",
            "submostboundid",
            "subhalfmass",
        ]

        dims = 3
        value = {}
        for i in range(self.ntask):
            name = self.basedir + "sub_tab_{0:03d}.{1}".format(self.num, i)
            ngroups = self.ngroups[i]
            nsubhalos = self.nsubhalos[i]

            data_types = {
                "nsubperhalo": "({0},)i4".format(ngroups),
                "firstsubofhalo": "({0},)i4".format(ngroups),

                "sublen": "({0},)i4".format(nsubhalos),
                "suboffset": "({0},)i4".format(nsubhalos),
                "subparenthalo": "({0},)i4".format(nsubhalos),

                "halo_m_mean200": "({0},)f4".format(ngroups),
                "halo_r_mean200": "({0},)f4".format(ngroups),
                "halo_m_crit200": "({0},)f4".format(ngroups),
                "halo_r_crit200": "({0},)f4".format(ngroups),
                "halo_m_tophat200": "({0},)f4".format(ngroups),
                "halo_r_tophat200": "({0},)f4".format(ngroups),

                "subpos": "({0},{1})f4".format(nsubhalos, dims),
                "subvel": "({0},{1})f4".format(nsubhalos, dims),
                "subspin": "({0},{1})f4".format(nsubhalos, dims),

                "subveldisp": "({0},)f4".format(nsubhalos),
                "subvmax": "({0},)f4".format(nsubhalos),
                "subhalfmass": "({0},)f4".format(nsubhalos),

                "submostboundid": "({0},)i8".format(nsubhalos),
            }
            with open(name, 'rb') as f:
                f.seek(5*4, 0)
                for key in array_keys:
                    dt = data_types[key]
                    data = fromfile(f, dtype=dt, count=1)[0]

                    if i == 0:
                        value[key] = data
                    else:
                        value[key] = concatenate([value[key], data])

        for key in array_keys:
            setattr(self, key, value[key])

    def _load_ids(self):
        basename = self.basedir
        basename += "sub_ids_{0:03d}.".format(self.num)

        for i in range(self.ntask):
            name = basename + "{0}".format(i)
            with open(name, 'rb') as f:
                f.seek(4*4, 0)
                nids = self.nids[i]
                ids = fromfile(f, dtype="i8", count=nids)
            if i == 0:
                all_ids = ids
            else:
                all_ids = concatenate(self.ids, ids)

        self.ids = []
        for sub in range(self.nsubhalos[0]):
            nmin = self.suboffset[sub]
            nmax = nmin + self.sublen[sub]
            self.ids.append(all_ids[nmin:nmax])

    def _locate_particles(self, subhalo):

        locations = {}
        for key in particle_keys:
            snap_ids = self.snap.read_block("id", key)
            if len(snap_ids) > 0:
                mask = snap_ids.isin(self.ids[subhalo])
                ind = snap_ids[mask].index
                locations[key] = ind

        return locations

    def _save_locations(self):

        print("Calculating locations")

        nsubhalos = 10

        ind = {}
        #with ProgressBar(nsubhalos) as bar:
        for subhalo in range(nsubhalos):
            location = self._locate_particles(subhalo)
            s = pd.Series(location)
            key = "%d" % subhalo
            ind[key] = s
        #       bar.update()

        store = pd.HDFStore(self.locationsname)
        for key in ind:
            store[key] = ind[key]
        store.close()

    def read_block_by_subhalo(self, block_type, particle_type, subhalo):

        block = self.snap.read_block(block_type, particle_type)

        sub_ids = self.ids[subhalo]

        return block.loc[sub_ids].dropna()
