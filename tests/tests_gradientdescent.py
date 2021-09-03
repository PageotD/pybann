import unittest
import numpy as np
from pybann import Model
from pybann import GradientDescent


class tests_gradientdescent_batchsize0(unittest.TestCase):

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

        # Create Gradient descent instance
        SGD = GradientDescent(dataset, 0, 0.05, 1000, 0.5, testModel.layers)

        self.assertEqual(SGD.alpha, 0.05)
        self.assertEqual(SGD.nepoch, 1000)
        self.assertEqual(SGD.momentum, 0.5)

        self.assertTupleEqual(SGD.dataset[0][0], dataset[0][0])
        self.assertTupleEqual(SGD.dataset[0][1], dataset[0][1])
        self.assertTupleEqual(SGD.dataset[1][0], dataset[1][0])
        self.assertTupleEqual(SGD.dataset[1][1], dataset[1][1])
        self.assertTupleEqual(SGD.dataset[2][0], dataset[2][0])
        self.assertTupleEqual(SGD.dataset[2][1], dataset[2][1])

    def test_initialize(self):

        # Create a dataset
        dataset = []
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
        dataset.append([(1., 1., 0., 0.), (1., 0., 0.)])
        dataset.append([(0., 1., 1., 0.), (0., 1., 0.)])
        dataset.append([(0., 0., 1., 1.), (0., 0., 1.)])
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

        # Create Gradient descent instance
        SGD = GradientDescent(dataset, 10, 0.05, 1000, 0.5, testModel.layers)

        self.assertEqual(SGD.alpha, 0.05)
        self.assertEqual(SGD.nepoch, 1000)
        self.assertEqual(SGD.momentum, 0.5)

        self.assertTupleEqual(SGD.dataset[0][0], dataset[0][0])
        self.assertTupleEqual(SGD.dataset[0][1], dataset[0][1])
        self.assertTupleEqual(SGD.dataset[1][0], dataset[1][0])
        self.assertTupleEqual(SGD.dataset[1][1], dataset[1][1])
        self.assertTupleEqual(SGD.dataset[2][0], dataset[2][0])
        self.assertTupleEqual(SGD.dataset[2][1], dataset[2][1])

    def test_initialize_update(self):

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

        # Create Gradient descent instance
        SGD = GradientDescent(dataset, 0, 0.05, 1000, 0.5, testModel.layers)
        SGD.init_update()

        self.assertEqual(np.sum(SGD.layers[1].weightsUpdate), 0.)
        self.assertEqual(np.sum(SGD.layers[2].weightsUpdate), 0.)
        self.assertEqual(np.sum(SGD.layers[1].biasesUpdate), 0.)
        self.assertEqual(np.sum(SGD.layers[2].biasesUpdate), 0.)

        self.assertEqual(np.sum(SGD.layers[1].weightsUpdateSave), 0.)
        self.assertEqual(np.sum(SGD.layers[2].weightsUpdateSave), 0.)
        self.assertEqual(np.sum(SGD.layers[1].biasesUpdateSave), 0.)
        self.assertEqual(np.sum(SGD.layers[2].biasesUpdateSave), 0.)

    def test_datasplit(self):

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

        # Create Gradient descent instance
        SGD = GradientDescent(dataset, 0, 0.05, 1000, 0.5, testModel.layers)
        inValues, outValues = SGD.datasplit(dataset[0])

        self.assertListEqual(list(inValues[0]), [1., 1., 0., 0.])
        self.assertListEqual(list(outValues[0]), [1., 0., 0.])

    def test_forward(self):

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

        # Create Gradient descent instance
        SGD = GradientDescent(dataset, 0, 0.05, 1000, 0.5, testModel.layers)
        inValues, outValues = SGD.datasplit(dataset[0])
        activation, transfer = SGD.forward(inValues)
        
        self.assertTupleEqual(np.shape(activation[0]), (4, 1))
        self.assertTupleEqual(np.shape(activation[1]), (8, 1))
        self.assertTupleEqual(np.shape(activation[2]), (3, 1))
        self.assertTupleEqual(np.shape(transfer[0]), (8, 1))
        self.assertTupleEqual(np.shape(transfer[1]), (3, 1))
