import os
import random

import numpy as np

random.seed(42)

os.system("g++ -std=c++11 heapu_test.cc -o heapu_test")

TESTS = 100

for t in range(10, TESTS):
    print("Running test #{:d}".format(t))
    n = t * t
    m = int(t * t * 0.5)
    wId = open("test.in", "w")
    wId.write("{:d} {:d}\n".format(n, m))
    answers = []
    heaps = dict()
    heapl = []
    deleted = set()
    cnt = 0
    for i in range(m):
        if random.random() < 0.5 or len(heapl) == 0:
            b = random.randint(0, n - 1)
            while b in deleted:
                b = random.randint(0, n - 1)
            c = random.randint(0, n - 1)
            wId.write("0 {:d} {:d}\n".format(b, c))
            if b in heaps:
                heapl.remove((heaps[b], b))
            heaps[b] = c
            heapl.append((heaps[b], b))
        else:
            tp = max(heapl)
            heapl.remove(tp)
            del(heaps[tp[1]])
            answers.append(tp[0])
            deleted.add(tp[1])
            cnt += 1
            wId.write("1\n")
    wId.close()
    answers = np.array(answers)
    os.system("heapu_test.exe < test.in > test.out")
    mat = np.loadtxt("test.out")
    if not np.array_equal(answers, mat):
        print("now you fucked up")
        print(answers, mat)
        print(cnt, len(answers), len(mat))
