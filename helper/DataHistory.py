
class DataHistory:
    def __init__(self):
        self.iterations = 0
        self.data = []

    def prepare_plot(self, y_value, iter_start=0, iter_end=-1):
        if iter_end == -1:
            iter_end = self.iterations

        x_plot_data = []
        y_plot_data = []

        for i in range(iter_start, iter_end):
            x_plot_data.append(i)
            y_plot_data.append(self.data[i][y_value])

        return x_plot_data, y_plot_data

    def add_record(self, record):
        self.data.append(record)
        self.iterations += 1
