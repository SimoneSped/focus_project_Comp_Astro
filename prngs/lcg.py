# Implementation of a Linear congruential generator from Wikipedia in form of a generator
# From https://en.wikipedia.org/wiki/Linear_congruential_generator#Python_code

from collections.abc import Generator

class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m