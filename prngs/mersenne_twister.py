
class MersenneTwister:
    def __init__(self, seed=None):
        self.w = 32  # word size
        self.n = 624  # degree of recurrence
        self.m = 397  # middle word, an offset
        self.a = 0x9908B0DF  # constant vector
        self.u = 11  # tempering bit shift
        self.s = 7   # tempering bit shift
        self.t = 15  # tempering bit shift
        self.l = 18  # tempering bit shift

        if seed is None:
            seed = 5489  # default seed
        self.init_genrand(seed)

    def init_genrand(self, seed):
        self.mt = [0] * self.n
        self.mt[0] = seed & ((1 << self.w) - 1)
        for i in range(1, self.n):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> (self.w - 2))) + i) & ((1 << self.w) - 1)