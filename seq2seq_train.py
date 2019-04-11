from seq2seq import encoder, decoder, Seq2Seq

def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch.loss = 0

    for batch, i in enumerate(iterator):
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

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            #trg = [len(trg sentence), batch_size]
            #output = [len(trg sentence), batch_size, output_dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



