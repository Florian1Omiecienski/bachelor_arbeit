#!/usr/bin/python3

import numpy as np
import math
import sys


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Invalid arguments: Specify str(path2entity_list) int(embedding_dim) str(out_path)")
        exit()
    
    entity_path = str(sys.argv[1])
    d = int(sys.argv[2])
    out_path = str(sys.argv[3])
    
    entities = []
    with open(entity_path) as ifile:
        for line in ifile:
            sline = line.strip(" \n")
            if sline == "":
                continue
            assert(sline[0]=="Q")
            entities.append(sline)
    n = len(entities)
    
    a = 1/math.sqrt(d)
    data = np.random.uniform(-a, a, (n,d))
    
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    with open(out_path, "tw") as ofile:
        for i in range(len(entities)):
            entity = entities[i]
            ofile.write("{} {}\n".format(entity, " ".join([str(f) for f in data[i]])))
