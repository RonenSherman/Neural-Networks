class Matrix:
    def __init__(self, data):
        """
         Matrix class without  WIP
        using - list of lists
        """
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix must be a list of lists.")
        if not data:
            self.rows = 0
            self.cols = 0
            self.data = []
            return
            # Note for self: Len means length
        self.rows = len(data)
        self.cols = len(data[0])
        if not all(len(row) == self.cols for row in data):
            raise ValueError("All rows in the matrix must have the same number of columns.")

        self.data = [list(row) for row in data]  # Create a copy to avoid mutation issues

    def __str__(self):
        """
        Returns a string representation of the matrix, so I don't have to make a "Print" function
        """
        return "\n".join([" ".join(map(str, row)) for row in self.data])

    def addition(self, other):
        """
        Performs matrix addition.
        """
        if not isinstance(other, Matrix) or self.rows != other.rows or self.cols != other.cols:
            raise ValueError("must have the same dimensions")

        result_data = []
        for r in range(self.rows):
            new_row = [self.data[r][c] + other.data[r][c] for c in range(self.cols)]
            result_data.append(new_row)
        return Matrix(result_data)

    def multiplication(self, other):
        """
         matrix multiplication
        """
        if isinstance(other, (int, float)):  # Scalar multiplication
            result_data = []
            for r in range(self.rows):
                new_row = [self.data[r][c] * other for c in range(self.cols)]
                result_data.append(new_row)
            return Matrix(result_data)

        elif isinstance(other, Matrix):  # Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(
                    "Number of columns in the first matrix must be the same as the number of rows in the second")

            result_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result_data[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix(result_data)
        else:
            raise TypeError("the type is wrong")




# Example usage:
matrix_a = Matrix([[1, 2], [3, 4]])
matrix_b = Matrix([[5, 6], [7, 8]])

print("Matrix A:")
print(matrix_a)

print("\nMatrix B:")
print(matrix_b)


