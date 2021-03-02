
class DataHistory:
    def __init__(self):
        self.iterations = 0
        self.data = []

    def prepare_plot(self, y_value, iter_start=0, iter_end=-1):
        if iter_end == -1:
            iter_end = self.iterations

        plot_data = []

        for i in range(iter_start, iter_end):
            plot_data[i].x = i
            plot_data[i].y = self.data[i][y_value]

        return plot_data

    def add_record(self, record):
        self.data[self.iterations] = record
        self.iterations += 1
