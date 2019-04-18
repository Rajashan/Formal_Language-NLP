import torch
import pandas as pd
import csv
import os


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.optimizer.zero_grad()
        
        output = model(src, trg[:,:-1])
        
                
        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]
            
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1])
            
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]
            
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)