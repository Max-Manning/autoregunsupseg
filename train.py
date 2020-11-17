import numpy as np
import torch 
from tqdm import tqdm
import pickle
from scipy.optimize import linear_sum_assignment as linear_assignment

from dataloader import Potsdam, PotsdamDataLoader
from model import ARSegmentationNet3, init_weights
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
    
    # get validation dataloader: using 'labelled test' split
    validation_dataset = Potsdam(path, split='labelled_test', is_test=True)
    validation_loader = PotsdamDataLoader(validation_dataset, batch_size=10)
    
    # define model, loss, optimzer, learning rate scheduler
    model = ARSegmentationNet3().to(device)
    model.apply(init_weights)
    criterion = MI_loss
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    orderings = np.arange(1,9)
    
    losses_train = []
    match_matrices = []
    
    epochs = 10
    for e in range(epochs):
        
        print(f"Starting epoch {e + 1}")
        print("Training ...")
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
            loss = criterion(out1, out2, 1) # with spatial invariance (T=1)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses_train.append(loss.item())
        
        confusion_matrix = np.zeros((3,3))
        print("Validating ...")
        
        ## VALIDATE ##
        for batch_idx, data in enumerate(tqdm(validation_loader)):
            with torch.no_grad():
                
                # get the predictions for the test data
                inputs = data[0].to(device)
                outputs = model(inputs, 0)
                
                # flatten the model predictions and ground truth labels
                labels = data[1].detach().numpy().flatten()
                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                
                # turn potsdam-6 labels into potsdam-3 labels
                labels[labels == 4] = 0 # merge road and cars classes
                labels[labels == 5] = 1 # merge buildings and clutter classes
                labels[labels == 3] = 2 # merge vegetation and trees classes
                
                # update the confusion matrix. Each pixel i has predicted label preds[i]
                # and ground truth label labels[i]
                for i in range(len(labels)):
                    confusion_matrix[preds[i], labels[i]] += 1
                    
        # use the Hungarian algorithm to find the best one-to-one mapping between predicted labels
        # and ground truth labels
        ri, ci = linear_assignment(confusion_matrix, maximize=True)
        
        # given the chosen mapping, how many pixels were correctly labeled?
        correct_labels = confusion_matrix[ri, ci].sum()
        
        # compute the pixel accuracy
        accuracy = correct_labels/confusion_matrix.sum()
        
        # save the confusion matrix for this epoch
        match_matrices.append(confusion_matrix)
            
        print(f"Epoch {e + 1} pixel accuracy: {accuracy*100:.2f} %")    
        
        # update lr
        scheduler.step()
    
    # save model weights and results
    f = open("saved/model_09_dist.pkl","wb")
    pickle.dump(match_matrices,f)
    f.close()
    
    torch.save(model.state_dict(), './saved/model_09.pth')
    
    losses_train = np.array(losses_train)
    np.save("./saved/losses_09.npy", losses_train)