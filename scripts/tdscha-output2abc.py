#!python
import sys, os
import numpy as np

MSG = """
TDSCHA  
======

Convert the output file of a Lanczos calculation 
into a .abc file to analyze easily the results.

Usage:

tdscha-output2abc.py  outputfile  abcfile

outputfile must contain the stdout of the Lanczos calculation
abcfile is generated from the output
"""


def convert(output, abcfile):
    reading = False
    index = 0

    a = []
    b = []
    c = []   
    with open(output, "r") as fp:
        for line in fp.readlines():
            line = line.strip()

            if "LANCZOS ALGORITHM" in line:
                reading = True

            data = line.split()
            if len(data) != 3:
                continue

            if "a_{:d}".format(index) == data[0]:
                a.append(float(data[2]))
            if "b_{:d}".format(index) == data[0]:
                b.append(float(data[2]))
            if "c_{:d}".format(index) == data[0]:
                c.append(float(data[2]))
                index += 1

    ERROR = """
Error, the length of a, b and c does not match!
len(a) = {}; len(b) = {}, len(c) = {}
""".format(len(a), len(b), len(c))
    
    assert len(a) == len(b), ERROR 
    assert len(a) == len(c), ERROR

    # Save the abc file
    np.savetxt(abcfile, np.transpose([a, b, c]))


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print(MSG)
        exit()
    
    if not os.path.exists(sys.argv[1]):
        raise IOError("""
Error, the file '{}' does not exist
""".format(sys.argv[1]))


    convert(sys.argv[1], sys.argv[2])
