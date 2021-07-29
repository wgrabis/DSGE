import numpy as np
from sympy import Matrix, pprint, diag
from scipy import linalg
import sympy as sp
import scipy

from helper.PolicyPrinter import PolicyPrinter


def c_inv(Z):
    if Z.det() == 0:
        return Z.pinv()
    return Z.inv()


class BlanchardRaw:
    def reorder_decomposition(self, F):
        # H, J = F.diagonalize()

        H = Matrix()

        size, _ = F.shape
        print("Reorder")
        pprint(F, wrap_line=False)
        print(size)

        eigen_vals = F.eigenvects(strict=False)
        print(eigen_vals)

        sorted_by_second = sorted(eigen_vals, key=lambda tup: abs(tup[0]))

        eigen_diag = []

        j = 0
        for eigen_val, multiplier, eigen_vectors in sorted_by_second:
            for i in range(multiplier):
                print(eigen_vectors[i])
                H = H.col_insert(j, eigen_vectors[i])

                # diag_vector = np.zeros(size)
                # diag_vector[j] = eigen_val
                j += 1

                eigen_diag.append(eigen_val)

                # J.append(diag_vector)

        H = Matrix(H)
        J = Matrix(np.diag(eigen_diag))

        print("Debug reorder")
        print(sorted_by_second)
        pprint(H)
        print("Diag")
        print(eigen_diag)
        pprint(J)
        pprint(F)
        pprint(H @ J @ H.inv())

        return H, J

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

        print("Debug matrices: F, G")
        pprint(F, wrap_line=False)
        pprint(G, wrap_line=False)

        H, J = self.reorder_decomposition(F)



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
        pprint(H, wrap_line=False)
        pprint(Hr, wrap_line=False)
        pprint(G, wrap_line=False)
        pprint(J, wrap_line=False)
        pprint(F11, wrap_line=False)
        pprint(F12, wrap_line=False)
        print("Debug HJHr")
        pprint(F, wrap_line=False)
        pprint(H @ J @ Hr, wrap_line=False)

        x_curr = x0

        x_values = []
        y_values = []

        print("transition function for x' : X, Y ,Shock")
        pprint(F11, wrap_line=False)
        pprint(F12, wrap_line=False)
        pprint(G1, wrap_line=False)

        y_transition = sp.re(-1 * H22R.pinv() @ H21R)
        y_shock = sp.re(-1 * H22R.pinv() @ J2.inv() @ (H21R @ G1 + H22R @ G2))

        x_transition = sp.re(F11 - F12 @ H22R.pinv() @ H21R)
        x_shock = sp.re(G1 - F12 @ H22R.pinv() @ J2.inv() @ (H21R @ G1 + H22R @ G2))

        print("transition function for y' : X ,Shock")
        pprint(sp.re(y_transition), wrap_line=False)
        pprint(sp.re(y_shock), wrap_line=False)

        print("Combined for x' from x, shock")
        pprint(sp.re(x_transition), wrap_line=False)
        pprint(sp.re(x_shock), wrap_line=False)

        print("Policy function")
        pprint(sp.re(x_transition), wrap_line=False)
        pprint(sp.re(y_transition), wrap_line=False)
        pprint(sp.re(y_shock), wrap_line=False)
        pprint(sp.re(x_shock), wrap_line=False)

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

    def singular_calculate(self, model, A, B, C, x0, shock, x_len, y_len, time):
        BB, AA, _, _, Q, Z = linalg.ordqz(B, A, sort=lambda a, b: pow(a, -1) * b, output="complex")

        print(AA)
        print(BB)

        print("QZ debug")
        pprint(Matrix(A), wrap_line=False)
        pprint(Matrix(B), wrap_line=False)
        pprint(Matrix(AA), wrap_line=False)
        pprint(Matrix(BB), wrap_line=False)
        pprint(Matrix(Q))
        pprint(Matrix(Z))
        print("DEBUG CHECK")
        pprint(Matrix(B), wrap_line=False)
        pprint(Q @ Matrix(BB) @ Z.T, wrap_line=False)
        print(x_len)

        print("SingularCalculate")
        pprint(B, wrap_line=False)
        pprint(Matrix(Q @ BB @ Z.T), wrap_line=False)
        pprint(Matrix(Q @ Q.T), wrap_line=False)
        print("AA, BB matrices:")
        pprint(Matrix(AA), wrap_line=False)
        pprint(Matrix(BB), wrap_line=False)
        pprint(c_inv(Matrix(AA)), wrap_line=False)
        print("EigenValues:")
        pprint(sp.re(c_inv(Matrix(AA)) @ BB), wrap_line=False)

        G = Q.T @ C

        reX0 = Z.T @ np.zeros(x_len + y_len)

        # X, Y = self.non_singular_calculate(Matrix(AA), Matrix(BB), Matrix(G), reX0[:x_len], shock, x_len, y_len, time)
        #
        # print("X, Y without inverse here")
        # pprint(X)
        # pprint(Y)

        AA = Matrix(AA)
        BB = Matrix(BB)
        Z = Matrix(Z)

        #todo continue

        Z11 = Z[:x_len, :x_len]
        Z12 = Z[:x_len, x_len:]
        Z21 = Z[x_len:, :x_len]
        Z22 = Z[x_len:, x_len:]

        ZT11 = Z.T[:x_len, :x_len]
        ZT12 = Z.T[:x_len, x_len:]
        ZT21 = Z.T[x_len:, :x_len]
        ZT22 = Z.T[x_len:, x_len:]

        T11 = AA[:x_len, :x_len]
        T12 = AA[:x_len, x_len:]
        T22 = AA[x_len:, x_len:]

        S11 = BB[:x_len, :x_len]
        S12 = BB[:x_len, x_len:]
        S22 = BB[x_len:, x_len:]

        #transition function

        g_y_plus = -1 * c_inv(Z22) @ Z21
        g_y_minus = ZT11 @ T11.inv() @ S11 @ c_inv(ZT11) #.inv()

        FY_plus = A[:, x_len:]
        FY = B - A
        FU = C

        FY_zero = sp.zeros(x_len + y_len, x_len + y_len)

        FY_zero[:, :x_len] = A[:, :x_len]
        FY_zero[:, x_len:] = -1 * B[:, x_len:]

        J_minus = sp.eye(x_len)

        for i in range(y_len):
            J_minus = J_minus.col_insert(x_len + i, sp.zeros(x_len, 1))

        print("Policy function")
        pprint(g_y_plus, wrap_line=False)
        pprint(g_y_minus, wrap_line=False)

        g_u = None

        print("Building policy for shock")
        print("FY_Plus")
        pprint(FY_plus, wrap_line=False)
        print("J_Minus")
        pprint(J_minus, wrap_line=False)
        print("GY+ part")
        pprint(FY_plus @ g_y_plus @ J_minus, wrap_line=False)
        print("FY_zero")
        pprint(FY_zero, wrap_line=False)
        print("Inverse part")
        if not x_len == 0:
            pprint((FY_plus @ g_y_plus @ J_minus + FY_zero).inv(), wrap_line=False)
        else:
            pprint(FY_zero.inv(), wrap_line=False)
        print("FU")
        pprint(FU, wrap_line=False)

        if x_len == 0:
            g_u = FY_zero.inv() @ FU
        else:
            g_u = (FY_plus @ g_y_plus @ J_minus + FY_zero).inv() @ FU

        g_u_plus = g_u[x_len:, :]
        g_u_minus = g_u[:x_len, :]

        print("Shock policy")
        pprint(g_u, wrap_line=False)
        pprint(g_u_plus, wrap_line=False)
        pprint(g_u_minus, wrap_line=False)

        PolicyPrinter.print(model, g_y_plus, g_y_minus, g_u)

        x_curr = x0

        x_values = []
        y_values = []

        for i in range(time):
            curr_shock = np.zeros(len(shock))

            if i == 0:
                curr_shock = shock

            x_next = g_y_minus @ x_curr + g_u_minus @ curr_shock
            y_next = g_y_plus @ x_curr + g_u_plus @ curr_shock

            print("Iter{}".format(i))
            print(x_next)
            print(y_next)

            x_curr = x_next

            x_values.append(x_next)
            y_values.append(y_next)

        return x_values, y_values






