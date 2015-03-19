#!/usr/bin/python
"""

"""
import numpy

from copy import deepcopy
from numpy import *

class Matrix(object):
	"""Represents a Matrix.
	
	Attributes:
		matrix: A list of lists for storing the matrix.
		rows: An integer for the number of rows in the matrix.
		columns: An integer for the number of columns in the matrix.
	"""
	
	def __init__(self, matrix, rows, columns):
		"""Inits the Matrix class with the matrix values."""
		self.__rows = rows
		self.__columns = columns
		self.__matrix = [[0 for x in range(columns)] for x in range(rows)]
		self.deepcopy_matrix(matrix)

	def get_matrix(self):
		'''Returns the matrix itself.
		'''
		return deepcopy(self.__matrix)

	def deepcopy_matrix(self, matrix):
		"""Copy the values of given matrix in the class matrix.

		Raises an exception if the dimensions of the input matrix 
		are not the same as that of the class matrix. Otherwise, all the
		elements are copied one by one.

		Args:
			matrix: The matrix from which the elements are copied.

		Raises:
			ValueError: Raised when dimensions mismatch.
		"""

		if not self.__rows == len(matrix) or ( len(matrix) and not self.__columns == len(matrix[0])):
			raise ValueError("Error: Dimensions do not match for both the matrices!!")

		for row in range(self.__rows):
			for col in range(self.__columns):
				self.__matrix[row][col] = matrix[row][col]

	def get(self, row, column):
		"""Return the value at the given position.

		For the given row and column, it returns the 
		corresponsing elements stored in the matrix.

		Args:
			row: Row number of the element to be returned.
			column: Column number of the element to be returned.

		Returns:
			The value stored at the given position.

		Raises:
			ValueError: Raised when given row and column are invalid.
		"""

		if row < 0 or row >= self.__rows or column < 0 or column >= self.__columns:
			raise ValueError("Error: Invalid row or column!!")

		return self.__matrix[row][column]

	def get_rows(self):
		"""Return the number of rows of the matrix.

		Returns:
			The number of rows in the matrix.
		"""
		return self.__rows

	def get_columns(self):
		"""Return the number of columns of the matrix.

		Returns:
			The number of columns in the matrix.
		"""
		return self.__columns

	def put_value(self, row, col, value):
		"""Change the value at given row and column.

		Args:
			row: Row number of the cell to be changed.
			col: Column number of the cell to be changed.
			value: Value to be finally put at that position.
		"""
		self.__matrix[row][col] = value

	def __add__(self, matrix):
		"""Add the matrices value by value.

		Adds each value of one matrix to the corresponsing
		value in the other matrix and returns the sum matrix.

		Args:
			matrix: The Matrix object to be added to this object.

		Returns:
			The sum matrix.

		Raises:
			ValueError: Raised when dimensions mismatch.
		"""	

		if not self.__rows == matrix.get_rows() or not self.__columns == matrix.get_columns():
			raise ValueError("Error: Dimensions do not match for both the matrices!!")

		sum_matrix = Matrix(deepcopy(self.__matrix), self.__rows, self.__columns)
		for row in range(self.__rows):
			for col in range(self.__columns):
				sum_matrix.put_value(row, col, self.__matrix[row][col] + matrix.get(row, col))

		return sum_matrix

	def __sub__(self, matrix):
		"""Subtracts one matrix from the other value by value.

		Subtracts each value of one matrix from the corresponsing
		value in the other matrix and returns the difference matrix.

		Args:
			matrix: The Matrix object to be subtracted from this object.

		Returns:
			The difference matrix.

		Raises:
			ValueError: Raised when dimensions mismatch.
		"""	

		if not self.__rows == matrix.get_rows() or not self.__columns == matrix.get_columns():
			raise ValueError("Error: Dimensions do not match for both the matrices!!")

		diff_matrix = Matrix(deepcopy(self.__matrix), self.__rows, self.__columns)
		for row in range(self.__rows):
			for col in range(self.__columns):
				diff_matrix.put_value(row, col, self.__matrix[row][col] - matrix.get(row, col))

		return diff_matrix

	def scalar_multiply(self, value):
		"""Multiplies each value of one matrix by the given value.

		Multiples each value of one matrix from the given
		value and returns the difference matrix.

		Args:
			value: The value which all the elements of the matrix are
				multiplied with.

		Returns:
			The multiplied matrix.
		"""	

		multiplied_matrix = Matrix(deepcopy(self.__matrix), self.__rows, self.__columns)
		for row in range(self.__rows):
			for col in range(self.__columns):
				multiplied_matrix.put_value(row, col, self.__matrix[row][col] * value)

		return multiplied_matrix

	def transpose(self):
		"""Return the transpose of a matrix.

		Returns:
			The transpose of the matrix.
		"""

		transposed_matrix = Matrix([[0 for x in range(self.__rows)] for x in range(self.__columns)], self.__columns, self.__rows)
		for row in range(transposed_matrix.get_rows()):
			for col in range(transposed_matrix.get_columns()):
				transposed_matrix.put_value(row, col, self.__matrix[col][row])

		return transposed_matrix

	def multiply_matrix(self, matrix):
		"""Multiply one matrix from the other.

		Multiplies two matrices.

		Args:
			matrix: The Matrix object to be multiplied with this object.

		Returns:
			The multiplication matrix.

		Raises:
			ValueError: Raised when dimensions mismatch.
		"""	
		if not self.__columns == matrix.get_rows():
			raise ValueError("Error: Dimensions do not match for both the matrices!!")

		multiplication_matrix = Matrix([[0 for x in range(matrix.get_columns())] for x in range(self.__rows)], self.__rows, matrix.get_columns())

		for row in range(self.__rows):
			for col in range(matrix.get_columns()):
				value = 0
				for k in range(self.__columns):
					value += self.__matrix[row][k] * matrix.get(k, col)

				multiplication_matrix.put_value(row, col, value)

		return multiplication_matrix

	def print_matrix(self):
		"""Prints the complete matrix.

		Print the matrix in the format of proper rows and 
		columns with one space between each value in a row and each
		column on different lines.
		""" 

		for row in range(self.__rows):
			row_values = ""
			for col in range(self.__columns):
				row_values += str(self.__matrix[row][col]) + " "

			print row_values

	def is_square_matrix(self):
		"""Return true or false depending upon if the matrix is square or not.

		Returns:
			true/false: Boolean value telling if the matrix is square or not.
		"""
		return self.__rows == self.__columns

	def inverse_matrix(self):
		"""Returns the inverse of a matrix if it is invertible.

		Returns:
			inverse matrix of the object matrix.

		Raises:
			ValueError: Raised if it is invertible matrix.
		"""

		if not self.is_square_matrix():
			raise ValueError("The matrix is not square matrix!!")

		inverted_matrix = None
		#try:
		inv_matrix = numpy.linalg.inv(self.__matrix)
		inverted_matrix = Matrix(inv_matrix.tolist(), self.__rows, self.__columns)
		#except:
		#	raise Exception("The matrix is singular!!")

		return inverted_matrix

	def feature_normalize(self):
		"""Normalize each feature.

		First, for each feature dimension, compute the mean 
		of the feature and subtract it from the dataset, 
		storing the mean value in mu. Next, compute the 
		standard deviation of each feature and divide
		each feature by it's standard deviation, storing
		the standard deviation in sigma.

		Returns:
			The new matrix object which is normalized. 
		"""
		mu = [0] * self.__columns
		sigma = [0] * self.__columns

		new_matrix = deepcopy(self.__matrix)
		new_matrix = numpy.array(new_matrix)

		for col in range(self.__columns):
			mu[col] = numpy.mean(new_matrix[:, col], axis = 0)
			sigma[col] = numpy.std(new_matrix[:, col], axis = 0)

			new_matrix[:, col] = (1/sigma[col]) * (new_matrix[:, col] - mu[col])

		normalized_matrix = Matrix(new_matrix.tolist(), self.__rows, self.__columns)
		return (normalized_matrix, mu, sigma)

if __name__ == '__main__':
	pass