import unittest
import numpy as np
from pybann import Layer

class tests_layers(unittest.TestCase):

    def test_initialize(self):

        testLayer = Layer(neurons=10, label='input')

        self.assertEqual(testLayer.neurons, 10)
        self.assertEqual(testLayer.label, 'input')

    def test_initialize_empty_params(self):

        testLayer = Layer(neurons=4)        

        self.assertEqual(testLayer.neurons, 4)
        self.assertEqual(testLayer.label, "")

    def test_addActivation_empty(self):

        testLayer = Layer(neurons=4)  
        testLayer.addActivation()

        self.assertEqual(testLayer.activation.__name__, "sigmoid")
        self.assertEqual(testLayer.activation(0), 0.5)
    
    def test_addActivation(self):

        testLayer = Layer(neurons=4)  
        testLayer.addActivation("tanhyp")

        self.assertEqual(testLayer.activation.__name__, "tanhyp")
        self.assertEqual(testLayer.activation(0), 0.)

    def test_addBiases(self):

        testLayer = Layer(neurons=4) 

        testLayer.addBiases()

        self.assertEqual(len(testLayer.biases), testLayer.neurons)
        self.assertEqual(len(testLayer.biasesUpdate), testLayer.neurons)

    def test_addWeights(self):

        testLayer = Layer(neurons=4) 

        testLayer.addWeights(inputNeurons=8)

        self.assertTupleEqual(np.shape(testLayer.weights), (8, 4))
        self.assertTupleEqual(np.shape(testLayer.weightsUpdate), (8, 4))