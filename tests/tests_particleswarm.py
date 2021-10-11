import unittest
import numpy as np
from pybann import Model
from pybann import ParticleSwarm


class tests_particleswarm(unittest.TestCase):

    def test_initialize(self):

        # Create a dataset
        dataset = []
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])

        # Create a model
        testModel = Model()

        # Add Layers
        testModel.addInput(neurons=4)
        testModel.addLayer(neurons=8)
        testModel.addLayer(neurons=3)

        # Build
        testModel.build()

        # Create PSO instance
        PSO = ParticleSwarm(dataset, 0, 0.4, 2.1, 2.1, 'full', 200, 1000, testModel.layers)

        self.assertEqual(PSO.omega, 0.4)
        self.assertEqual(PSO.cog, 2.1)
        self.assertEqual(PSO.soc, 2.1)
        self.assertEqual(PSO.topology, 'full')
        self.assertEqual(PSO.nepoch, 1000)
        self.assertEqual(PSO.nparticles, 200)

        self.assertTupleEqual(PSO.dataset[0][0], dataset[0][0])
        self.assertTupleEqual(PSO.dataset[0][1], dataset[0][1])
        self.assertTupleEqual(PSO.dataset[1][0], dataset[1][0])
        self.assertTupleEqual(PSO.dataset[1][1], dataset[1][1])
        self.assertTupleEqual(PSO.dataset[2][0], dataset[2][0])
        self.assertTupleEqual(PSO.dataset[2][1], dataset[2][1])

    def test_init_particles(self):

        # Create a dataset
        dataset = []
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])

        # Create a model
        testModel = Model()

        # Add Layers
        testModel.addInput(neurons=4)
        testModel.addLayer(neurons=8)
        testModel.addLayer(neurons=3)

        # Build
        testModel.build()

        # Create PSO instance
        PSO = ParticleSwarm(dataset, 0, 0.4, 2.1, 2.1, 'full', 200, 1000, testModel.layers)
        PSO.init_particles()