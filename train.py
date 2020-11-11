import numpy as np
import torch 
from tqdm import tqdm

from dataloader import Potsdam, PotsdamDataLoader
from model import ARSegmentationNet2, init_weights
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
    train_dataset = Potsdam(path, split=['unlabelled_train', 'labelled_train'])
    training_loader = PotsdamDataLoader(train_dataset, batch_size=10)
    
#     # get validation dataloader
#     validation_dataset = Potsdam(path, split='labelled_train')
#     validation_loader = PotsdamDataLoader(train_dataset, batch_size=8)
    
    # define model, loss, optimzer, learning rate scheduler
    model = ARSegmentationNet2().to(device)
    model.apply(init_weights)
    criterion = MI_loss
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    orderings = np.arange(1,9)
    
    losses = []
    K = np.logspace(-3, 1, 10)
    
    epochs = 10
    for e in range(epochs):
        
        print(f"Starting epoch {e + 1}")
        
        ## TRAIN ##
        for batch_idx, data in enumerate(tqdm(training_loader)):
            
            inputs = data.to(device)
            
            # randomly choose two orderings
            o1 = np.random.choice(orderings)
            o2 = np.random.choice(orderings)
            
            # compute the model outputs using each of the orderings
            out1 = model(inputs, o1)
            out2 = model(inputs, o2)
            
            # compute the MI loss between the two outputs
            # loss = criterion(out1, out2) # no spatial invariance (T=0)
            loss = criterion(out1, out2, 0) # with spatial invariance (T=1)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        # update lr
        scheduler.step()
    
    torch.save(model.state_dict(), './model_02.pth')
    
    losses = np.array(losses)
    np.save("losses_02.npy", losses)