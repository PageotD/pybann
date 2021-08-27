import unittest
import os
import numpy as np
from pybann import Model

class tests_model(unittest.TestCase):

    def test_initialize_empty(self):

        testModel = Model()

        self.assertEqual(testModel.name, "New model")
        self.assertTrue(hasattr(testModel, "layers"))

    def test_initialize(self):

        testModel = Model(name="Test model")

        self.assertEqual(testModel.name, "Test model")
        self.assertTrue(hasattr(testModel, "layers"))

    def test_addInput_empty(self):

        testModel = Model(name="Test model")

        testModel.addInput(neurons=8)

        self.assertEqual(testModel.layers[0].neurons, 8)
        self.assertEqual(testModel.layers[0].label, "")

    def test_addInput(self):

        testModel = Model(name="Test model")

        testModel.addInput(neurons=8, label="input")

        self.assertEqual(testModel.layers[0].neurons, 8)
        self.assertEqual(testModel.layers[0].label, "input")

    def test_addLayer_empty(self):

        testModel = Model(name="Test model")

        testModel.addLayer(neurons=8)

        self.assertEqual(testModel.name, "Test model")
        self.assertEqual(len(testModel.layers), 1)

        self.assertEqual(testModel.layers[0].neurons, 8)
        self.assertEqual(testModel.layers[0].activation.__name__, "sigmoid")
        self.assertEqual(testModel.layers[0].label, "")

    def test_addLayer(self):

        testModel = Model(name="Test model")

        testModel.addLayer(neurons=8, activation="tanhyp", label="input")

        self.assertEqual(testModel.name, "Test model")
        self.assertEqual(len(testModel.layers), 1)

        self.assertEqual(testModel.layers[0].neurons, 8)
        self.assertEqual(testModel.layers[0].activation.__name__, "tanhyp")
        self.assertEqual(testModel.layers[0].label, "input")
    
    def test_build(self):

        testModel = Model(name="Test model")

        # Create a (3, 5, 4) model
        testModel.addInput(neurons=3)
        testModel.addLayer(neurons=5)
        testModel.addLayer(neurons=4)
        testModel.build()

        self.assertTupleEqual(np.shape(testModel.layers[1].biases), (5, 1))
        self.assertTupleEqual(np.shape(testModel.layers[1].weights), (5, 3))

        self.assertTupleEqual(np.shape(testModel.layers[-1].biases), (4, 1))
        self.assertTupleEqual(np.shape(testModel.layers[-1].weights), (4, 5))

    def test_forward(self):

        np.random.seed(0)

        testModel = Model(name="Test model")

        # Create a (3, 5, 4) model
        testModel.addInput(neurons=3)
        testModel.addLayer(neurons=5)
        testModel.addLayer(neurons=4)
        testModel.build()

        inValues = [1., 1., 1.]
        outValues = testModel.forward(inValues)
        attemptedOutput = [0.33773496, 0.49363117, 0.94101299, 0.0350716]
        
        self.assertEqual(len(outValues), 4)
        for i in range(4):
            self.assertAlmostEqual(outValues[i], attemptedOutput[i], delta=1.e-7)

    def test_save(self):

        np.random.seed(0)

        testModel = Model(name="Test model")

        # Create a (3, 5, 4) model
        testModel.addInput(neurons=3)
        testModel.addLayer(neurons=5)
        testModel.addLayer(neurons=4)
        testModel.build()

        testModel.save("tests_testsave")
        self.assertTrue(os.path.isfile("tests_testsave"))

    def test_load(self):

        np.random.seed(0)

        testModel = Model(name="Test model")

        # Create a (3, 5, 4) model
        testModel.addInput(neurons=3)
        testModel.addLayer(neurons=5)
        testModel.addLayer(neurons=4)
        testModel.build()

        testModel.save("tests_testsave")
        testLoadModel = Model(name="Test model")
        testLoadModel.load("tests_testsave")

        self.assertTrue(isinstance(testLoadModel, Model))
        self.assertEqual(testModel.name, testLoadModel.name)