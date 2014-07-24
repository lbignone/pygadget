# Author: Lucas A. Bignone
# Contact: lbignone@iafe.uba.ar


from __future__ import division

from pygadget import Simulation, Subfind
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import pandas as pd

import gc


# Set a function that returns the redshift for a given look-back time
model = FlatLambdaCDM(H0=70, Om0=0.3)
z = np.linspace(0.0, 20.0, 1000)
t = model.lookback_time(z)

redshift_from_time = interp1d(t.value, z)


def analytical_mass_metallicity(z, stellar_mass):
    """Analytical mass-metallicty relationship as computed by Maiolino 2008

    Args:
        z: redshift, valid values are 0.07, 0.7, 2.2 and 3.5
        stellar_mass: stellar mass in MSun

    Returns:
        Oxygen abundance in 12 + log10(O/H)
    """

    observational_parameters = {
        0.07: {"logM0": 11.18, "K0": 9.04},
        0.7: {"logM0": 11.57, "K0": 9.04},
        2.2: {"logM0": 12.38, "K0": 8.99},
        3.5: {"logM0": 12.76, "K0": 8.79},
    }

    logM0 = observational_parameters[z]['logM0']
    K0 = observational_parameters[z]['K0']

    a = -0.0864

    result = a * (np.log10(stellar_mass) - logM0)**2 + K0

    return result


def metallicity_correction(z, Z, a, b):
    """Metallicty correction to the simulation metal yields

    Args:
        z: redshift
        Z: metallcity
        a: free parameter

    Returns:
        corrected metallicity
    """

    result = Z * 10**(a*z + b)

    return result


def oxygen_abun_from_metallicity(df):
    """Compute 12 + log10(O/H) from snapshot metallicty

    Args:
        df: dataframe containing oxygen (O) an hydrogen (H) columns
        as returned from Simulation.read_block("metals", ...)

    Returns:
        Oxygen abundance in 12 + log10(O/H)
    """
    NOxygen = df.O / 16.0
    NHydrogen = df.H

    S = 12.0 + np.log10(NOxygen / NHydrogen)
    S.name = '12 + log(O/H)'
    S.replace(-np.inf, np.nan, inplace=True)

    return S


def mass_inside_radius(df, radius):
    """ Compute mass inside radius

    Args:
        df: dataframe containing a 'mass' and a 'r' (radius) columns
        redius: radius in witch to compute the mass

    Returns:
        mass inside redius
    """

    dframe = df[df.r < radius]

    mass_inside = (dframe.mass).sum()
    return mass_inside


def simulated_mass_metallicity(sub):
    """ Compute mass-metallicty relationship for simulations

    Args:
        sub: pygadget Subfind object with associated snapshot

    Returns:
        (mass, metallicity) tuple
    """

    # filter all subhalos with less than 1000 particles
    subhalos = np.where(sub.sublen >= 1000)[0]

    sub_stellar_mass = []
    sub_stellar_metallicity = []
    for subhalo in subhalos:

        optical_radius = sub.optical_radius(subhalo)
        used_radius = 1.5 * optical_radius

        # read blocks
        mass = sub.read_block_by_subhalo("mass", "stars", subhalo)
        pos = sub.read_block_by_subhalo("pos", "stars", subhalo)
        stellar_metallicity = sub.read_block_by_subhalo("metals",
                                                        "stars", subhalo)

        # compute radius
        cm = sub.subpos[subhalo]
        r = np.sqrt(((pos-cm)**2).sum(axis=1))
        r.name = 'r'

        # load redshifts
        redshift = sub.read_block_by_subhalo("age", "stars", subhalo)
        redshift = 1./redshift - 1.0

        df = pd.concat([redshift, r, mass, stellar_metallicity], axis=1)

        stellar_mass = mass_inside_radius(df, used_radius)

        # Use only stars younger than 100 Myr
        z_min = sub.snap.redshift
        t0 = model.lookback_time(z_min)
        t1 = t0 + 10 * u.Myr
        z_max = redshift_from_time(t1).max()

        df2 = df[df.age <= z_max]
        df2 = df2[df2.r <= used_radius]

        oxygen_abundance = oxygen_abun_from_metallicity(df2)
        oxygen_abundance = oxygen_abundance.mean()

        if (stellar_mass > 0.0):
            if (not np.isnan(oxygen_abundance)):
                # append to final list
                sub_stellar_mass.append(stellar_mass)
                sub_stellar_metallicity.append(oxygen_abundance.mean())

    sub_stellar_mass = np.array(sub_stellar_mass)
    sub_stellar_metallicity = np.array(sub_stellar_metallicity)
    return sub_stellar_mass, sub_stellar_metallicity


def main():
    """Compute the corrections necessary to reproduce the mass-metallicity
    relationship of galaxies at every redshift.

    The correction assumed is:
    [12 + log(O/H)]_corrected = [12 + log(O/H)] + a*z +b

    where z is redshift and (a, b) are the parameters to be computed
    """

    basedir = "/home/lbignone/simulations/MARE_230_D"

    snap_numbers = [37, 24, 10, 8]
    redshifts = []
    redshifts2 = [0.07, 0.7, 2.2, 3.5]
    mass = {}
    metallicty = {}

    colors = colormap.rainbow(np.linspace(0, 1, len(snap_numbers)))

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                                 figsize=(12, 7))

    for snap, color in zip(snap_numbers, colors):

        snap_name = basedir + "/snap_230_{0:03d}".format(snap)

        snapshot = Simulation(snap_name, pot=True)

        redshifts.append(snapshot.redshift)

        sub = Subfind(basedir, snap, snapshot)

        mass[snap], metallicty[snap] = simulated_mass_metallicity(sub)

        label = "{0:.1f}".format(snapshot.redshift)
        ax1.scatter(np.log10(mass[snap]*1e10), metallicty[snap], alpha=0.5,
                    color=color, label=label)

        # keeps memory usage down by releasing objects
        del snapshot
        del sub
        gc.collect()

    ax1.set_title("uncorrected")
    ax1.set_ylabel("12 + log(O/H)")
    ax1.legend(loc='best')

    # Use the results at redshift 0 to fit the b parameter
    def func1(mass, b):
        model = analytical_mass_metallicity(0.07, mass)
        sim = b
        return model - sim

    p0 = [0]
    popt, pvar = curve_fit(func1, mass[37]*1e10, metallicty[37], p0=p0)
    b = popt[0]

    # Use the results at redshift 3.5 to fit the a parameter
    def func2(mass, a):
        model = analytical_mass_metallicity(3.5, mass)
        sim = a*3.5 + b
        return model - sim

    p0 = [0]
    popt, pvar = curve_fit(func2, mass[8]*1e10, metallicty[8], p0=p0)
    a = popt[0]

    print("a: {0}".format(a))
    print("b: {0}".format(b))

    sm = np.linspace(7.0, 11.0, 1000)
    for snap, color, redshift, r2 in zip(snap_numbers,
                                         colors, redshifts, redshifts2):
        metals = metallicty[snap] + (a*redshift + b)

        label = "{0:.1f}".format(redshift)

        plt.scatter(np.log10(mass[snap]*1e10), metals, alpha=0.5,
                    color=color, label=label)
        ax2.plot(sm, analytical_mass_metallicity(r2, 10**sm), color=color)

    ax2.set_title("corrected")
    ax2.set_xlabel("log(M* [MSun])")
    ax2.set_ylabel("12 + log(O/H)")
    ax2.legend(loc='best')
    ax2.set_ylim(4.0, 10.0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
