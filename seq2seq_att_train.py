import torch
import pandas as pd
import csv
import os

def trainer(model, iterator, optimizer, criterion, clip):
    
    model.train()

    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_len, trg)
        
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()

    if os.path.exists('seq2seq_output.csv'):
        os.remove('seq2seq_output.csv')
    if os.path.exists('seq2seq_trg.csv'):
        os.remove('seq2seq_trg.csv')
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output, _ = model(src, src_len, trg, 0) #turn off teacher forcing

            output_out = torch.argmax(output, dim = 2)
            
            trg_out = trg
        
            #iterate over batch size
            for j in range(len(trg_out[1,:])):
                #
                with open('seq2seq_trg.csv', 'a') as file1:
                    file1.write(pd.DataFrame(trg_out[:,j].cpu().numpy()).T.to_string(index = False, header = False))
                    file1.write(",")
                    #print(trg_out[:,j].numpy())
                    file1.close()
                #pd.DataFrame(trg_out[:,i].numpy()).to_csv("seq2seq_trg.csv", header=None, index=None)
                with open('seq2seq_output.csv', 'a') as file2:
                    file2.write(pd.DataFrame(output_out[:,j].cpu().numpy()).T.to_string(index = False, header = False))
                    file2.write(",")
                    file2.close()

            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)