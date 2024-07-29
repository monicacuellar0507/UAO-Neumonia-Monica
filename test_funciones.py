import unittest
import tensorflow as tf
import load_model

class TestModelFun(unittest.TestCase):
    def test_model_loading(self):
        # Comprueba si el modelo se carga correctamente
        loaded_model = load_model.model_fun()
        self.assertIsInstance(loaded_model, tf.keras.models.Model, "El modelo debe ser una instancia de tf.keras.models.Model")

if __name__ == '__main__':
    unittest.main()