import numpy as np
import torch 

from dataloader import Potsdam, PotsdamDataLoader
from model import ARSegmentationNet
from loss import MI_loss


if __name__ == "__main__":
    
    '''SIMPLE TRAINING SCRIPT'''
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")

    # get the dataloader
    path = '/mnt/D2/Data/potsdam/preprocessed/'
    train_dataset = Potsdam(path, split='unlabelled_train')
    training_loader = PotsdamDataLoader(train_dataset, batch_size=5)
    
    # get validation dataloader
    # validation_dataset = Potsdam(path, split='labelled_train')
    # validation_loader = PotsdamDataLoader(train_dataset, batch_size=20)
    
    # define model, loss, optimzer, learning rate scheduler
    model = ARSegmentationNet().to(device)
    criterion = MI_loss
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    orderings = np.arange(1,9)
    
    # start training loop
    epochs = 1
    for e in range(epochs):
        
        print(f"Starting epoch {e}")
        
        for batch_idx, data in enumerate(training_loader):
            
            print(f"Training on batch number {batch_idx}")
            
            inputs = data.to(device)
            
            # changed the dataloader so it doesn't return labels
            # will need to fix this later so I can do validation
            # inputs = data[0].to(device)
            # labels = data[1].to(device)
            
            # randomly choose two orderings
            o1 = np.random.choice(orderings)
            o2 = np.random.choice(orderings)
            
            # compute the model outputs using each of the orderings
            out1 = model(inputs, o1)
            out2 = model(inputs, o2)
            
            # compute the MI loss between the two outputs
            loss = criterion(out1, out2)
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # update lr
        scheduler.step()