

class PosteriorStory:
    def __init__(self):
        self.posteriors = []
        self.burnout = 0.4
        self.len = 0

    def add(self, posterior, distribution):
        self.len += 1
        self.posteriors.append((posterior, distribution))

    def get_post_burnout(self):
        return self.posteriors[int(self.len * self.burnout):]

    def last(self):
        return self.posteriors[self.len - 1]
