import unittest

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../'))

from src.matrix import Matrix

class TestMatrix(unittest.TestCase):
	'''Test matrix class.

	It contains unit test cases for all the functionalities
	of the matrix class.
	'''
	def setUp(self):
		'''Setup for the test.'''
		self.matrix_normal = Matrix([[1,2,3],[4,5,6]], 2, 3)
		self.matrix_one_dimensional = Matrix([[1,2,3]], 1, 3)
		self.matrix_square = Matrix([[1,2,3], [4,5,6], [7,8,9]],3,3)
		self.empty_matrix = Matrix([],0,0)

	def test_constructor_matrix(self):
		'''Test the constructor of the matrix.'''
		super(TestMatrix, self).assertListEqual(self.matrix_normal.get_matrix(), [[1,2,3],[4,5,6]])
		super(TestMatrix, self).assertListEqual(self.matrix_one_dimensional.get_matrix(), [[1,2,3]])
		super(TestMatrix, self).assertListEqual(self.matrix_square.get_matrix(), [[1,2,3],[4,5,6],[7,8,9]])
		super(TestMatrix, self).assertListEqual(self.empty_matrix.get_matrix(), [])

	def test_add_matrix(self):
		'''Test the add matrices function in matrix class. '''
		super(TestMatrix, self).assertRaises(ValueError, self.matrix_normal.__add__, self.matrix_one_dimensional)
		matrix_to_be_added = Matrix([[1,2,3],[4,5,6]], 2, 3)
		final_matrix = Matrix([[2,4,6],[8,10,12]], 2, 3)
		super(TestMatrix, self).assertEqual((self.matrix_normal + matrix_to_be_added).get_matrix(), final_matrix.get_matrix())

	def test_sub_matrix(self):
		'''Test the sub matrices function in matrix class. '''
		super(TestMatrix, self).assertRaises(ValueError, self.matrix_normal.__sub__, self.matrix_one_dimensional)
		matrix_to_be_subtracted = Matrix([[1,2,3],[4,5,6]], 2, 3)
		final_matrix = Matrix([[0,0,0],[0,0,0]], 2, 3)
		super(TestMatrix, self).assertListEqual((self.matrix_normal - matrix_to_be_subtracted).get_matrix(), final_matrix.get_matrix())

	def test_scalar_multiply(self):
		'''Test the scalar multiply function in matrix class. '''
		scalar_value = 2
		super(TestMatrix, self).assertListEqual(self.matrix_normal.scalar_multiply(scalar_value).get_matrix(), [[2,4,6], [8, 10, 12]])
		super(TestMatrix, self).assertListEqual(self.matrix_one_dimensional.scalar_multiply(scalar_value).get_matrix(), [[2,4,6]])
		super(TestMatrix, self).assertListEqual(self.empty_matrix.scalar_multiply(scalar_value).get_matrix(), [])

	def test_transpose(self):
		'''Test the transpose of a matrix function in matrix class. '''
		super(TestMatrix, self).assertListEqual(self.matrix_normal.transpose().get_matrix(), [[1, 4], [2, 5], [3, 6]])
		super(TestMatrix, self).assertListEqual(self.matrix_one_dimensional.transpose().get_matrix(), [[1], [2], [3]])
		super(TestMatrix, self).assertListEqual(self.empty_matrix.transpose().get_matrix(), [])

	def test_multiply_matrices(self):
		'''Test the multiplication of matrices.'''
		test_matrix1 = Matrix([[12,7,3], [4,5,6], [7,8,9]], 3, 3)
		test_matrix2 = Matrix([[5,8,1,2], [6,7,3,0], [4,5,9,1]], 3, 4)
		super(TestMatrix, self).assertListEqual(test_matrix1.multiply_matrix(test_matrix2).get_matrix(), [[114, 160, 60, 27], [74, 97, 73, 14], [119, 157, 112, 23]])

		test_matrix1 = Matrix([[1,2,3]], 1, 3)
		test_matrix2 = Matrix([[1], [2], [3]], 3, 1)
		super(TestMatrix, self).assertListEqual(test_matrix1.multiply_matrix(test_matrix2).get_matrix(), [[14]])

	def test_inverse_matrix(self):
		"""Test the inverse of a matrix."""
		test_matrix_singular = Matrix([[1,2,3], [4,5,6], [3,4,5]], 3, 3)
		test_matrix_normal = Matrix([[8,2,3], [4,5,6], [3,4,5]], 3, 3)
		inverse_normal_matrix = Matrix([[0.1428, 0.2857, -0.4285], [-0.2857, 4.4285, -5.1428], [0.1428, -3.7142, 4.5714]], 3, 3)

		super(TestMatrix, self).assertRaises(Exception, test_matrix_singular.inverse_matrix)

		inverted_matrix = test_matrix_normal.inverse_matrix()
		for row in range(inverted_matrix.get_rows()):
			for col in range(inverted_matrix.get_columns()):
				super(TestMatrix, self).assertAlmostEqual(inverted_matrix.get_matrix()[row][col], inverse_normal_matrix.get_matrix()[row][col], 3)

	def test_feature_normalize(self):
		"""Test the feature normalization of a matrix."""
		test_matrix = Matrix([[1,2,3], [5,6,7]], 2, 3)
		normalized_matrix, mu, sigma = test_matrix.feature_normalize()

		super(TestMatrix, self).assertListEqual(mu, [3,4,5])
		super(TestMatrix, self).assertListEqual(sigma, [2,2,2])
		super(TestMatrix, self).assertListEqual(normalized_matrix.get_matrix(), [[-1, -1, -1], [1, 1, 1]])

def suite():
	'''Returns a suite with instances of TestMatrix.
	Each instance tests the method indicated by its constructor.'''
	suite = unittest.TestSuite()
	suite.addTest(TestMatrix("test_constructor_matrix"))
	suite.addTest(TestMatrix("test_add_matrix"))
	suite.addTest(TestMatrix("test_sub_matrix"))
	suite.addTest(TestMatrix("test_scalar_multiply"))
	suite.addTest(TestMatrix("test_transpose"))
	suite.addTest(TestMatrix("test_multiply_matrices"))
	suite.addTest(TestMatrix("test_feature_normalize"))
	return suite

if __name__ == '__main__':
	unittest.main()