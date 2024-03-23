import os
import sys
from tqdm import tqdm
import time



def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()

if __name__ == '__main__':
    start_time = time.time()
    threshold = str(sys.argv[1])
    path ='Cov/activeneuron/' + threshold + 'ase/'
    cov = readFile(path + 'neuron_cov')
    cnum = len(cov[0])
    nnum = len(cov)
    f = open(path + 'test_cov','w')
    for i in tqdm(range(cnum)):
        tstr = ''
        for j in range(nnum):
            if cov[j][i] == '1':
                tstr += '1'
            else:
                tstr += '0'
        f.write(tstr + '\n')
    f.close()

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Save the execution time to a file
    with open("execution_time_2process.txt", "w") as file:
        file.write("Execution time: {} seconds".format(execution_time))

    print("Execution time:", execution_time, "seconds")

