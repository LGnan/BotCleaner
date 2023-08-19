import numpy as np
from fractions import Fraction

vector1 = np.array([16, 7])
vector2 = np.array([15, 5])

distancia = np.linalg.norm(vector1 - vector2)
redondo = round(distancia,2)
fraction = Fraction(redondo)
rise, run = fraction.numerator, fraction.denominator
print(redondo)
print(rise, run)

numerator = 7-5
denominator = 16 - 15

quotient = numerator // denominator
remainder = numerator % denominator

print(quotient, remainder)