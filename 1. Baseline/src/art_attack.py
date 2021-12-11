import time
import torch
import numpy as np

def test_art(dataloader, classifier, loss_fn, attack=None):
    start_time = time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0.0, 0
    for batch, (data, label) in enumerate(dataloader):
        if attack is not None:
            data = data.numpy()
            data = attack.generate(data)
        pred = classifier.predict(data)
        val_loss += loss_fn(torch.from_numpy(pred), label).item()
        correct += np.sum(np.argmax(pred, axis=1) == label.detach().numpy())
        if batch%10 == 0:
            current = batch*len(data)
            print(f'[{current:4d}/{size:4d} = {(100*current/size):4.1f}%], batch {batch}, time: {(time.time() - start_time):0.1f} s')
    val_loss /= num_batches
    correct /= size
    print(f'Val Error: \n Accuracy: {(100*correct):0.6f}%, Avg loss: {val_loss:0.6f}')
    return correct, val_loss