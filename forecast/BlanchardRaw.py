import numpy as np
from sympy import Matrix, pprint
from scipy import linalg
import scipy


class BlanchardRaw:
    def reorder_decomposition(self, F):
        # H, J = F.diagonalize()

        H = Matrix()
        J = []

        size, _ = F.shape
        print("Reorder")
        print(size)

        eigen_vals = F.eigenvects()
        print(eigen_vals)

        sorted_by_second = sorted(eigen_vals, key=lambda tup: abs(tup[0]))

        j = 0
        for eigen_val, multiplier, eigen_vectors in sorted_by_second:
            for i in range(multiplier):
                print(eigen_vectors[i])
                H = H.col_insert(j, eigen_vectors[i])

                diag_vector = np.zeros(size)
                diag_vector[j] = eigen_val
                j += 1

                J.append(diag_vector)


        print("Debug reorder")
        print(sorted_by_second)
        pprint(Matrix(H))
        pprint(Matrix(J))
        pprint(F)
        pprint(Matrix(H) @ Matrix(J) @ Matrix(H).inv())

        return Matrix(H), Matrix(J)

    def non_singular_calculate(self, A, B, C, x0, shock, x_len, y_len, time):
        # AA, BB, Q, Z = scipy.linalg.qz(A, B)
        #
        # pprint("Blanchard raw")
        # print(AA)
        # pprint(Q)
        # pprint(Z)
        # pprint(A)
        # pprint(Q @ AA @ Z.T)
        # pprint(B)
        # pprint(Q @ BB @ Z.T)
        #
        # x_0 = Z * x0

        Ainv = A.inv()

        F = Ainv * B
        G = Ainv * C

        H, J = self.reorder_decomposition(F)

        print("Debug matrices: F, G")
        pprint(F)
        pprint(G)

        # check blanchard kahn condition

        Hr = H.inv()

        # H11R = Hr[:x_len, :x_len]
        # H12R = Hr[:x_len, x_len:]
        H21R = Hr[x_len:, :x_len]
        H22R = Hr[x_len:, x_len:]

        F11 = F[:x_len, :x_len]
        F12 = F[:x_len, x_len:]

        # J1 = J[:x_len, :x_len]
        J2 = J[x_len:, x_len:]

        G1 = G[:x_len, :]
        G2 = G[x_len:, :]

        print("Debug matrices: H, G, J, F11, F12")
        pprint(H)
        pprint(Hr)
        pprint(G)
        pprint(J)
        pprint(F11)
        pprint(F12)
        print("Debug HJHr")
        pprint(F)
        pprint(H @ J @ Hr)

        x_curr = x0

        x_values = []
        y_values = []

        print("transition function for x' : X, Y ,Shock")
        pprint(F11)
        pprint(F12)
        pprint(G1)
        print("transition function for y' : X ,Shock")
        pprint(-1 * H22R.inv() @ H21R)
        pprint(-1 * H22R.inv() @ J2.inv() @ (H21R @ G1 + H22R @ G2))

        print("Combined for x' from x, shock")
        pprint((F11 - F12 @ H22R.inv() @ H21R))
        pprint((G1 - F12 @ H22R.inv() @ J2.inv() @ (H21R @ G1 + H22R @ G2)))

        y_transition = -1 * H22R.inv() @ H21R
        y_shock = -1 * H22R.inv() @ J2.inv() @ (H21R @ G1 + H22R @ G2)

        x_transition = (F11 - F12 @ H22R.inv() @ H21R)
        x_shock = (G1 - F12 @ H22R.inv() @ J2.inv() @ (H21R @ G1 + H22R @ G2))

        for i in range(time):
            x_state_part = x_transition @ x_curr
            curr_shock = np.zeros(len(shock))

            if i == 0:
                curr_shock = shock

            x_next = x_state_part + x_shock @ curr_shock
            y_next = y_transition @ x_curr + y_shock @ curr_shock

            print("Iter{}".format(i))
            print(x_next)
            print(y_next)

            x_curr = x_next

            x_values.append(x_next)
            y_values.append(y_next)

        return x_values, y_values







