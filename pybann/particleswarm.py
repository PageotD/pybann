"""
Particle Swarm Optimization
"""

from typing import Union
import numpy as np
from tqdm import tqdm
from copy import deepcopy

class ParticleSwarm:
    """
    Class for the paticle swarm optimization (PSO) method.
    *Standard* *barebone* and *fully-informed* declinaison of PSO are available
    with the *inertia weight* and the *constriction factor* strategies. Several
    topologies are lso implemented: *full*, *ring* and *toroidal*.
    """

    def __init__(self, dataset, batchsize: int, omega: int, cog: float, soc: float,
                 topology: str, nparticles: int, nepoch: int, layers) -> None:

        self.dataset = dataset
        self.nepoch = nepoch
        self.layers = layers
        self.batchsize = batchsize
        self.omega = omega
        self.cog = cog
        self.soc = soc
        self.topology = topology
        self.nparticles = nparticles 

        self.current = None
        self.velocity = None
        self.history = None
        self.misfit = None
        self.pspace = None

    def init_particles(self):

        self.particles = []
        
        for iparticle in range(self.nparticles):
            # Create a copy of the neural network
            tmp = deepcopy(self.layers)
            # Redifine initial weights and biases
            for ilayer in range(1, len(self.layers)):
                tmp[ilayer].weights = np.random.randn(*np.shape(tmp[ilayer].weights))
                tmp[ilayer].biases = np.random.randn(*np.shape(tmp[ilayer].biases))
            # Add
            self.particles.append(tmp)

    def run(self):
        """
        Train
        """
        self.init_particles()

    def _get_neighbors(self, topology, indv, ndim):
        """
        Return an array containing the indices of the neighbors particles.

        :param indv: indice of the particle to update
        :param ndim: number of particles in the first dimension if toroidal grid is used
        """

        # Get the number of particles
        nindv = self.current.shape[0]

        # Full topology (including the particle itself)
        if topology == 'full':
            neighborhood = np.zeros(nindv, dtype=np.int)
            for i in range(0, nindv):
                neighborhood[i] = i

        # Ring topology (including the particle itself)
        if topology == 'ring':
            neighborhood = np.zeros(3, dtype=np.int)
            ineighbor = 0
            for i in range(indv-1, indv+2):
                neighborhood[ineighbor] = i
                if i < 0:
                    neighborhood[ineighbor] = nindv-1
                if i == nindv:
                    neighborhood[ineighbor] = 0
                ineighbor += 1

        # Toroidal topology (including the particle itself)
        if topology == 'toroidal':
            # If the number of particles is a multiple of ndim
            if nindv%ndim == 0:
                # Get grid size
                n1 = ndim
                n2 = int(nindv/ndim)
                neighborhood = np.zeros(5, dtype=np.int)
                # Get the indice of the particle on the grid
                i2 = int(indv/ndim)
                i1 = int(indv-i2*n1)
                # Fill neighborhood
                neighborhood[0] = indv
                # Get the indice of the neighbors
                # top
                if i1 == 0:
                    neighborhood[1] = i2*n1+(n1-1)
                else:
                    neighborhood[1] = i2*n1+(i1-1)
                # right
                if i2 == n2-1:
                    neighborhood[2] = i1
                else:
                    neighborhood[2] = (i2+1)*n1+i1
                # bottom
                if i1 == n1-1:
                    neighborhood[3] = i2*n1
                else:
                    neighborhood[3] = i2*n1+(i1+1)
                # left
                if i2 == 0:
                    neighborhood[4] = ((n2-1)*n1)+i1
                else:
                    neighborhood[4] = (i2-1)*n1+i1

        return neighborhood

    def _get_grid(self, ndim):
        """
        Define toroidal grid

        :param ndim: number of particles in the first dimension of the toroidal grid.
        """

        # Get the number of particles
        nindv = self.current.shape[0]

        # If the number of particles is a multiple of ndim
        if nindv%ndim == 0:
            # Initialize grid dimensions
            n1 = ndim
            n2 = int(nindv/ndim)
            # Initialize neighbor array (4 neighbor per particle)
            vngrid = np.zeros((nindv, 4), dtype=np.int)
            # Loop over toroidal grid dimensions
            for i2 in range(0, n2):
                for i1 in range(0, n1):
                    # Get the indice of the neighbors
                    k = (i2*n1)+i1
                    # top
                    if i1 == 0:
                        vngrid[k, 0] = i2*n1+(n1-1)
                    else:
                        vngrid[k, 0] = i2*n1+(i1-1)
                    # right
                    if i2 == n2-1:
                        vngrid[k, 1] = i1
                    else:
                        vngrid[k, 1] = (i2+1)*n1+i1
                    # bottom
                    if i1 == n1-1:
                        vngrid[k, 2] = i2*n1
                    else:
                        vngrid[k, 2] = i2*n1+(i1+1)
                    # left
                    if i2 == 0:
                        vngrid[k, 3] = ((n2-1)*n1)+i1
                    else:
                        vngrid[k, 3] = (i2-1)*n1+i1

        # The number of particles is not a multiple of ndim
        else:
            raise ValueError('ndim must be a multiple of nindv')

        return vngrid

    def get_gbest(self, topology, indv=0, ndim=0):
        """
        Get gbest particle of the whole swarm or in the neighborhood of
        a given particle.

        .. rubric:: Basic usage

        >>> topology = 'full'
        >>> best = population.get_gbest(topology)
        """

        nindv = self.current.shape[0]

        # Get the best particle of the whole swarm
        if topology == 'full':
            ibest = np.argmin(self.misfit[:])

        # Get the best particle in the neighborhood (1 left, 1 right)
        # of the particle including itself.
        if topology == 'ring':
            ibest = indv
            vbest = self.misfit[indv]
            for i in range(indv-1, indv+2):
                ii = i
                if i < 0:
                    ii = nindv-1
                if i == nindv:
                    ii = 0
                if self.misfit[ii] < vbest:
                    ibest = ii
                    vbest = self.misfit[ii]

        # Get the best particle in the neighborhood (1 left, 1 right)
        # of the particle excluding itself.
        if topology == 'ringx':
            ileft = indv-1
            iright = indv+1
            if indv == 0:
                ileft = nindv-1
            if indv == nindv-1:
                iright = 0
            if self.misfit[ileft] <= self.misfit[iright]:
                ibest = ileft
            else:
                ibest = iright

        # Get the best particle in the neighborhood (left, right, top, bottom)
        # of the particle including itself.
        if topology == 'toroidal':
            grid = self._get_grid(ndim)
            ibest = indv
            vbest = self.misfit[indv]
            for i in range(0, 4):
                if self.misfit[grid[indv, i]] < vbest:
                    ibest = grid[indv, i]
                    vbest = self.misfit[grid[indv, i]]

        # Get the best particle in the neighborhood (left, right, top, bottom)
        # of the particle excluding itself.
        if topology == 'toroidalx':
            # Get the grid
            grid = self._get_grid(ndim)
            # Initialize best particule to a virtual particle with a maximum misfit
            ibest = -1
            vbest = np.amax(self.misfit[:]*2.)
            for i in range(0, 4):
                if self.misfit[grid[indv, i]] < vbest:
                    ibest = grid[indv, i]
                    vbest = self.misfit[grid[indv, i]]

        return self.history[ibest, :, :]

    def update(self, **kwargs):
        """
        Standard PSO update.

        :param control: 0 for weight (default), 1 for constriction
        :param c_0: value of the control parameter (default 0.7298)
        :param c_1: value of the cognitive parameter (default 2.05)
        :param c_2: value of the social parameter (default 2.05)
        :param topology: used topology (default 'full'): full, ring, ringx, toroidal, toroidalx
        :param ndim: number of particles in the first dimension if toroidal topology is used
        :param pupd: parameter update probability
        """

        # Parse kwargs parameter list
        ctrl = kwargs.get('control', 0)
        omega = kwargs.get('c_0', 0.7298)
        topology = kwargs.get('topology', 'full')
        ndim = kwargs.get('ndim', 0)
        pupd = kwargs.get('pupd', 2.0)

        if ctrl == 0:
            cog = kwargs.get('c_1', 2.05)
            soc = kwargs.get('c_2', 2.05)
        if ctrl == 1:
            cog = omega*kwargs.get('c_1', 2.05)
            soc = omega*kwargs.get('c_2', 2.05)

        # Update process
        for indv in range(0, self.current.shape[0]):
            gbest = self.get_gbest(topology, indv, ndim)
            for ipts in range(0, self.pspace.shape[0]):
                # Test if parameter will be updated
                if np.random.random_sample() <= pupd:
                    for ipar in range(0, self.pspace.shape[1]):

                        # Get values
                        current = self.current[indv, ipts, ipar]
                        velocity = self.velocity[indv, ipts, ipar]
                        history = self.history[indv, ipts, ipar]

                        # Update velocity vector
                        self.velocity[indv, ipts, ipar] = omega*velocity\
                                                    + cog*np.random.random_sample()\
                                                    * (history-current)\
                                                    + soc*np.random.random_sample()\
                                                    * (gbest[ipts, ipar]-current)

                        # Check particle velocity
                        if(np.abs(self.velocity[indv, ipts, ipar]) > self.pspace[ipts, ipar, 2]):
                            self.velocity[indv, ipts, ipar] = \
                                np.sign(self.velocity[indv, ipts, ipar])\
                                * self.pspace[ipts, ipar, 2]

                        # Update particle position
                        self.current[indv, ipts, ipar] += self.velocity[indv, ipts, ipar]

                        # Check if particle is in parameter space
                        if(self.current[indv, ipts, ipar] < self.pspace[ipts, ipar, 0]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 0]
                        if(self.current[indv, ipts, ipar] > self.pspace[ipts, ipar, 1]):
                            self.current[indv, ipts, ipar] = self.pspace[ipts, ipar, 1]

    def run(self):
        """
        Train the artificial neural network using particle swarm optimization
        """
        pass