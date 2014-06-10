from struct import unpack
from numpy import fromstring


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

    particle_keys = [
        "gas",
        "halo",
        "disk",
        "buldge",
        "stars",
        "bndry",
    ]

    block_keys = [
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

        if block_type in ["pos", "vel", "accel", "pot", "tstp"]:
            particle_number = self.particle_numbers
            if block_type in ["pos", "vel", "accel"]:
                dim = 3

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
