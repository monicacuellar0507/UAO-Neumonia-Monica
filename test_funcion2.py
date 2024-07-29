import unittest
import numpy as np
import preprocess_img

class TestPreprocess(unittest.TestCase):
    def test_preprocess(self):
        # se crea una imagen de ejemplo
        image = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

        # se llama la función preprocess de la clase detector_neumonia
        preprocessed_image = preprocess_img.preprocess(image)

        # se valida si la forma de salida es correcta
        self.assertEqual(preprocessed_image.shape, (1, 512, 512, 1))

        # se comprueba si los valores de píxeles están dentro del rango esperado
        self.assertTrue(np.all(preprocessed_image >= 0) and np.all(preprocessed_image <= 1))

if __name__ == "__main__":
    unittest.main()
