import sympy as sym

class EquationNormalize:
    @staticmethod
    def normalize(A, B, C):
        row_count, col_count = A.shape
        _, shock_count = C.shape

        Aout = sym.zeros(row_count, col_count)
        Bout = sym.zeros(row_count, col_count)
        Cout = sym.zeros(row_count, shock_count)

        print("Debug normalize")

        for i in range(row_count):
            a_row = A[i, :]
            b_row = B[i, :]
            c_row = C[i, :]

            print("{} row".format(i))
            print(a_row)
            print(b_row)

            positive_signed = 0

            for j in range(col_count):
                if a_row[0, j] >= 0:
                    positive_signed += 1
                if b_row[0, j] >= 0:
                    positive_signed += 1

            if positive_signed < col_count:
                a_row = -1 * a_row
                b_row = -1 * b_row
                c_row = -1 * c_row

            Aout[i, :] = a_row[0, :]
            Bout[i, :] = b_row[0, :]
            Cout[i, :] = c_row[0, :]

        return Aout, Bout, Cout


