import torch
import pandas as pd
import csv
import os


def trainer(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src,trg)

        #trg = [len(trg sentence), batch_size]
        #output = [len(trg sentence), batch_size, output_dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        #trg = [(len(trg sentence - 1)) * batch_size]
        #output = [(len(trg sentence - 1)) * batch_size, output_dim]

        loss = criterion(output,trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):

    model.eval()
    #should find more elegant way
    if os.path.exists('seq2seq_output.csv'):
        os.remove('seq2seq_output.csv')
    if os.path.exists('seq2seq_trg.csv'):
        os.remove('seq2seq_trg.csv')

    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #print(i)
            src = batch.src
            trg = batch.trg
            

            output = model(src, trg, 0)
            #greedy decode
            output_out = torch.argmax(output, dim = 2)
            trg_out = trg
            
            for j in range(len(trg_out[1,:])):
                
                with open('seq2seq_trg.csv', 'a') as file1:
                    file1.write(pd.DataFrame(trg_out[:,j].numpy()).T.to_string(index = False, header = False))
                    file1.write(",")
                    #print(trg_out[:,j].numpy())
                    file1.close()
                #pd.DataFrame(trg_out[:,i].numpy()).to_csv("seq2seq_trg.csv", header=None, index=None)
                with open('seq2seq_output.csv', 'a') as file2:
                    file2.write(pd.DataFrame(output_out[:,j].numpy()).T.to_string(index = False, header = False))
                    file2.write(",")
                    file2.close()
            
                #pd.DataFrame(output_out[:,i].numpy()).to_csv("seq2seq_output.csv", header=None, index=None)
            
            #trg = [len(trg sentence), batch_size]
            #output = [len(trg sentence), batch_size, output_dim]

            output = output[1:].view(-1, output.shape[-1])
            
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)



