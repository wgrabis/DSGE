from sympy import pprint


class BlanchardRaw:
    def calculate(self, A, B, C, x0, shock, x_len, y_len):
        A_inv = A.pinv()

        F = A_inv * B
        G = A_inv * C

        H, J = F.diagonalize()

        HR = H.inv()

        H11R = HR[:x_len, :x_len]
        H12R = HR[:x_len, x_len:]
        H21R = HR[x_len:, :x_len]
        H22R = HR[x_len:, x_len:]

        G1 = G[:x_len, :]
        G1 = G[x_len:, :]

        F11 = F[:x_len, :x_len]
        F12 = F[:x_len, x_len:]

        J1 = J[0:x_len, 0:x_len]
        J2 = J[x_len:, x_len:]

        print("H, J, F, G")
        pprint(H, wrap_line=False)
        pprint(J, wrap_line=False)
        pprint(F, wrap_line=False)
        pprint(G, wrap_line=False)