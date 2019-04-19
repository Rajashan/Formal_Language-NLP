import pandas as pd
import numpy as np
from distance_metrics import lcs
# wei zhengchang de seq2seq, bie de dou you tese de wenti

trg = pd.read_csv("seq2seq_trg.csv", sep=",",header = None)
trg = trg.T 

trg = trg[0].str.replace(" ","")
trg = trg[~trg.astype(float).isnull()]
trg = [s[1:] for s in trg]


output = pd.read_csv("seq2seq_output.csv",sep = ",",header = None)
output = output.T
output = output[0].str.replace(" ","")
output = output[~output.astype(float).isnull()]
output = [s[:] for s in output] 

LCS = [lcs.llcs(v,u) for v,u in zip(trg,output)]

print("debug ahoy!")

loss = 0

for row in range(len(output)):
    if output[row] == trg[row]:
        None
    else:
        loss += 1 
        
match = loss / len(trg)
