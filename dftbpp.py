import numpy as np
import os

class mycoords:
    """ Object to accumulate coordinates, each in an (natoms x nsteps) array"""
    def __init__(self, x, y, z, name, time=None, vels=None):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.nframes = x.shape[1]
        self.natoms = x.shape[0]
        if time is not None:
            self.time = time
        if vels is not None:
            self.vels = vels
        
    def centercoords(self):
        """Center the coordinates"""
        self.x = self.x - self.x.sum(axis=0)/self.x.shape[0]
        self.y = self.y - self.y.sum(axis=0)/self.y.shape[0]
        self.z = self.z - self.z.sum(axis=0)/self.z.shape[0]
        
    def dist(atom1,atom2,step):
        return np.sqrt((self.x[atom1,step] - self.x[atom2,step])**2 + \
                       (self.y[atom1,step] - self.y[atom2,step])**2 + (self.z[atom1,step] - self.z[atom2,step])**2)
        
    def radius(self, atom, time):
        return np.sqrt((self.x[atom, time])**2 + (self.y[atom, time])**2 + (self.z[atom, time])**2)

    def esfradius(self):
        """Calculates the radius of the outer sphere (longest distance to the center)"""
        r = np.sqrt((self.x - self.x[0,:])**2 + (self.y - self.y[0,:])**2 + (self.z - self.z[0,:])**2)
        return abs(r).max(axis=0)

    def getsurfatoms(self,r):
        rads = {55:4.47, 147:6.9, 309:9}
        surfats = []
        for i in range(self.natoms):
            if r[i,0] > rads[self.natoms]:
                surfats.append(i)
        return surfats
    
    def rad_esferas(self,surfatoms=None):
        """ Calcula el radio vs t de la esfera más chica capaz de contener a la nanopartícula (radext), \
            y el radio vs t de la esfera más grande contenida por los átomos superficiales (radint)"""
        r = np.sqrt((self.x - self.x[0,:])**2 + (self.y - self.y[0,:])**2 + (self.z - self.z[0,:])**2)
        surfatoms = self.getsurfatoms(r)
        rsurfall = np.array([r[i,:] for i in surfatoms])
        radext = abs(r).max(axis=0)
        radint = rsurfall.min(axis=0)
        return radext, radint
    
    def rad_promedio(self):
        """
        Weighted average of radius of all surface atoms (<= than rad_esferas).
        """
        r = np.sqrt((self.x - self.x[0,:])**2 + (self.y - self.y[0,:])**2 + (self.z - self.z[0,:])**2)
        surfatoms = self.getsurfatoms(r)
        rsurfall = np.array([r[i,:] for i in surfatoms])
        return rsurfall.sum(axis=0)/len(surfatoms)
    
    def printoneframe(self, frame, filename):
        """ Prints one snapshot of coordinates. Hardcoded for silver for the moment. """
        with open(filename, 'a') as out:
            out.write(str(self.natoms)+'\n')
            out.write('\n')
            for at in range(self.natoms):
                out.write('Ag  {:15.8f}  {:15.8f}  {:15.8f}\n'.format(\
                self.x[at,frame], self.y[at,frame], self.z[at,frame]))
                  
    def printcoords(self, printevery, savetomanyfiles=False):
        """ Prints coordinates to file every many frames """
        #newframes = int(self.nframes/printevery)
        if not savetomanyfiles:
            if os.path.isfile('tdcoords.pp.xyz'):
                os.remove('tdcoords.pp.xyz')
        for idx,frame in enumerate(range(0, self.nframes, printevery)):
            #idx = frame*printevery
            if savetomanyfiles:
                os.mkdir('frame'+str(idx))
                self.printoneframe(frame, 'frame'+str(idx)+'/coords.xyz')
            else:
                self.printoneframe(frame, 'tdcoords.pp.xyz')
    
class myenergies:
    """Object to accumulate different componentes of the energy"""
    def __init__(self, tot, nonscc, scc, ext, rep, kin):
        self.tot = tot   
        self.nonscc = nonscc
        self.scc = scc
        self.ext = ext
        self.rep = rep
        self.kin = kin
        
def getcoords_MD(file):
    """Function to read the coordinates from a BOMD run and generate a mycoords object containing them"""

    md = open(file,'r')
    md = md.readlines()
    natoms = int(md[0].strip().split()[0])
    nframes = int(len(md)/(natoms+2))
    x = [0]*natoms
    y = [0]*natoms
    z = [0]*natoms
    
    for i in range(natoms):
        x[i] = []
        y[i] = []
        z[i] = []

    for i in range(nframes):
        line = (natoms+2)*i+2
        for atom in range(0,natoms):
            x[atom].append(float(md[line+atom].strip().split()[1]))
            y[atom].append(float(md[line+atom].strip().split()[2]))
            z[atom].append(float(md[line+atom].strip().split()[3]))

    return mycoords(np.array(x), np.array(y), np.array(z))


def getcoords(file, bomd=False, veloc=False):
    """Function to read the coordinates and generate a mycoords object containing them"""
    md = open(file,'r')
    md = md.readlines()
    natoms = int(md[0].strip().split()[0])
    x = [0]*natoms
    y = [0]*natoms
    z = [0]*natoms
    names = []
    if not bomd:
        time = []
    for i in range(natoms):
        x[i] = []
        y[i] = []
        z[i] = []
    if veloc:    
        vx = [0]*natoms
        vy = [0]*natoms
        vz = [0]*natoms
        for i in range(natoms):
            vx[i] = []
            vy[i] = []
            vz[i] = []
    
    for i in range(len(md)):
        if 'MD' in md[i]:
            if not bomd:
                time.append(float(md[i].strip().split()[4]))
            if i==1:
                names, thiscoords = readCoords(md, natoms, names, MDline=i)
            else:
                thiscoords = readCoords(md, natoms, names, MDline=i)
            for atom in range(natoms):
                x[atom].append(thiscoords[0][atom])
                y[atom].append(thiscoords[1][atom])
                z[atom].append(thiscoords[2][atom])
            if veloc:
                for atom in range(1,natoms+1):
                    vx[atom-1].append(float(md[i+atom].strip().split()[4]))
                    vy[atom-1].append(float(md[i+atom].strip().split()[5]))
                    vz[atom-1].append(float(md[i+atom].strip().split()[6]))
    
    if not bomd:
        if veloc:
            return mycoords(np.array(x), np.array(y), np.array(z), names, np.array(time), \
            np.array([vx, vy, vz]))
        else:
            return mycoords(np.array(x), np.array(y), np.array(z), names, np.array(time)) 
    else:
        return mycoords(np.array(x), np.array(y), np.array(z), names)

    
def readCoords(thisfile, natoms, names, MDline=1):
    """Read one set of coordinates """ 
    coords = [[],[],[]]
    for atom in range(1,natoms+1):
        if MDline == 1:
            names.append(thisfile[MDline+atom].strip().split()[0])
        coords[0].append(float(thisfile[MDline+atom].strip().split()[1]))
        coords[1].append(float(thisfile[MDline+atom].strip().split()[2]))
        coords[2].append(float(thisfile[MDline+atom].strip().split()[3]))
    if MDline == 1:
        return names, coords
    else:
        return coords


def getenergies(filename):
    """Function to read the energy componentes and generate a myenergies object containing them"""
    data = np.genfromtxt(filename)
    time = data[:,0]
    etot = data[:,1] # etot  
    enonscc = data[:,2] #e non scc
    escc = data[:,3] # e scc
    eext = data[:,5] # e ext
    erep = data[:,6] # e rep
    ekin = data[:,7] # e kin
    return time, myenergies(etot, enonscc, escc, eext, erep, ekin)


def cutArrays(array, maximo):
    """Function to cut the array length for bond arrays and histogram arrays"""
    if array[:, maximo:].all() == 0. or not array[:, maximo:]:       # Note aux is the last one calculated
        newArray = array[:, :maximo]
        return newArray
    else:
        print('Number of bonds in range has changed! Check range')

    
def importDat(filename, saveevery, emin, emax):
    """Function to import energy per bond files when written in ASCII format"""
    with open(filename) as f:
        natomssqr = len(f.readline().split()) - 2 
        nlines = sum(1 for line in f)

    time = np.zeros((int(nlines/saveevery)+1))
    bonds = np.zeros((int(nlines/saveevery)+1, natomssqr))
 
    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            if i % saveevery == 0:
                index = int(i/saveevery)
                a = line.split()
                aux = [float(x) for x in a[2:] if (abs(float(x)) < emax and abs(float(x)) > emin)]
                bonds[index,:len(aux)] = aux[:]
                time[index] = float(a[0])
    
    bonds = cutArrays(bonds, len(aux))  
    return time, bonds


def importBin(filename, saveevery, emin, emax, natoms, nlines):
    """Function to import energy per bond files when written in binary format (unformatted),
    selecting only energies between emin and emax"""
    f = open(filename,'rb')
    field = np.fromfile(f,dtype='float64',count=(natoms**2+2)*(nlines+1))
    field = np.reshape(field,((nlines+1),(natoms**2+2)))
    f.close()
    
#    nlines = field.shape[0]
    time = np.zeros((nlines))
    bonds = np.zeros((nlines, natoms**2))
    
    for i in range(nlines):
        index = i #int(i/saveevery)
        aux = [float(x) for x in field[i,2:] if (abs(float(x)) < emax and abs(float(x)) > emin)]
        bonds[index,:len(aux)] = aux[:]
        time[index] = field[i,0]

    bonds = cutArrays(bonds, len(aux))  
    return time, bonds

def importBinAll(filename, saveevery, natoms, nlines):
    """Function to import energy per bond files when written in binary format (unformatted)"""
    f = open(filename,'rb')
    field = np.fromfile(f,dtype='float64',count=(natoms**2+2)*(nlines+1))
    field = np.reshape(field,((nlines+1),(natoms**2+2)))
    f.close()
    time = np.zeros((nlines))
    bonds = np.zeros((nlines, natoms, natoms))
    
    for i in range(nlines):
        aux = np.array([float(x) for x in field[i,2:]])
        bonds[i,:len(aux)] = aux.reshape(natoms,natoms)
        time[i] = field[i,0]
    return time, bonds


def createHistogram(bonds, binsize, binmin=None, binmax=None):
    """Function to create histograms from the bonds array"""
    hists = np.zeros_like(bonds)
    bmin = min(bonds[0,:]) # a tiempo 0
    bmax = max(bonds[0,:]) # a tiempo 0
    if binmin is not None:
        bmin = binmin
    if binmax is not None:
        bmax = binmax

    bins = np.arange(bmin, bmax+binsize, binsize)
    for i in range(bonds.shape[0]):
        hist, binsidx = np.histogram(bonds[i,:], bins)
        hists[i,:len(hist)] = hist

    hists = cutArrays(hists, len(hist))
    return hists, binsidx, bins


