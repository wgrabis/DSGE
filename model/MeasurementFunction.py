import numpy as np


class MeasurementFunction:
    def __init__(self, m_base, m_time, m_state):
        self.m_base = m_base
        self.m_time = m_time
        self.m_state = m_state

    def __call__(self, state, time):
        v_state = np.dot(self.m_state, state)
        v_time = np.dot(self.m_time, time)

        print(self.m_state)
        print(self.m_time)
        print(state)
        print(time)
        print(v_state)
        print(v_time)
        print(self.m_base)

        return self.m_base + v_time + v_state
