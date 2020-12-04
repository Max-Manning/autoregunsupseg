import numpy as np
import torch 
from tqdm import tqdm
import pickle
import argparse
from scipy.optimize import linear_sum_assignment as linear_assignment

from datasets.dataloader_potsdam import Potsdam, PotsdamDataLoader
from datasets.dataloader_cocostuff import get_coco_dataloader
from model.model import ARSegmentationNet2, ARSegmentationNet2A, ARSegmentationNet3, ARSegmentationNet3A, ARSegmentationNet4, ARSegmentationNet4A, init_weights
from model.loss import MI_loss

def parse_args():

    parser = argparse.ArgumentParser(
        description="Autoregressive Unsupervised Image Segmentation")
    
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['Potsdam', 'Potsdam3', 'CocoStuff3', 'CocoStuff15'],
        help='''Which dataset to use. Choose from 'Potsdam', 'Potsdam3',
        'CocoStuff3' or 'CocoStuff15'. ''')
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='''Name for saving the output files''')
    parser.add_argument(
        '--batch_size',
        default=10,
        type=int,
        help='Batch size.')
    parser.add_argument(
        '--learning_rate',
        default=2e-5,
        type=float,
        help='Learning rate.')
    parser.add_argument(
        '--spatial_invariance',
        default=1,
        type=int,
        help='Spatial invariance for the MI loss.')
    parser.add_argument(
        '--attention',
        type=bool,
        default=False,
        help='Whether to use an attention layer.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs.')
    parser.add_argument(
        '--num_res_layers',
        type=int,
        default=2,
        choices=[1,2,3,4],
        help='Number of residual layers in the autoregressive encoder.')
    parser.add_argument(
        '--output_stride',
        type=int,
        default=2,
        choices=[2,4],
        help='Output stride for the convolutional stem')

    return parser.parse_args()

def main(ARGS):

    if ARGS.dataset == 'Potsdam':
        # get the dataloader
        path = '/mnt/D2/Data/potsdam/preprocessed/'
        train_dataset = Potsdam(path, coarse_labels=False, split=['unlabelled_train', 'labelled_train'])
        training_loader = PotsdamDataLoader(train_dataset, batch_size=ARGS.batch_size)
        # get validation dataloader: using 'labelled test' split
        validation_dataset = Potsdam(path, coarse_labels=False, split='labelled_test', is_test=True)
        validation_loader = PotsdamDataLoader(validation_dataset, batch_size=ARGS.batch_size)
        in_channels = 4
        num_classes = 6
        
    elif ARGS.dataset == 'Potsdam3':
        # get the dataloader
        path = '/mnt/D2/Data/potsdam/preprocessed/'
        train_dataset = Potsdam(path, coarse_labels=True, split=['unlabelled_train', 'labelled_train'])
        training_loader = PotsdamDataLoader(train_dataset, batch_size=ARGS.batch_size)
        # get validation dataloader: using 'labelled test' split
        validation_dataset = Potsdam(path, coarse_labels=True, split='labelled_test', is_test=True)
        validation_loader = PotsdamDataLoader(validation_dataset, batch_size=ARGS.batch_size)
        in_channels = 4
        num_classes = 3
        
    elif ARGS.dataset == 'CocoStuff15':
        base_path = '/mnt/D2/Data/CocoStuff164k/'
        training_loader = get_coco_dataloader(ARGS.batch_size, base_path, version='CocoStuff15', split='train')
        validation_loader = get_coco_dataloader(ARGS.batch_size, base_path,  version='CocoStuff15', split='val')
        in_channels = 3
        num_classes = 15
        
    elif ARGS.dataset == 'CocoStuff3':
        base_path = '/mnt/D2/Data/CocoStuff164k/'
        training_loader = get_coco_dataloader(ARGS.batch_size, base_path, version='CocoStuff3', split='train')
        validation_loader = get_coco_dataloader(ARGS.batch_size, base_path, version='CocoStuff3', split='val')
        in_channels = 3
        num_classes = 4 # this is confusing since it's called coco-stuff 3
        
    else:
        raise ValueError("""Incorrect dataset. Please choose one of:
                'Potsdam', 'Potsdam3', 'CocoStuff15', 'CocoStuff3'. """)
        
        
    if ARGS.output_stride == 2:
        conv1_stride=1
    else:
        conv1_stride=2
    
    if ARGS.attention:

        if ARGS.num_res_layers == 2:
            model = ARSegmentationNet2A(in_channels=in_channels, num_classes=num_classes).to(device)
        elif ARGS.num_res_layers == 3:
            model = ARSegmentationNet3A(in_channels=in_channels, num_classes=num_classes).to(device)
        elif ARGS.num_res_layers == 4:
            model = ARSegmentationNet4A(in_channels=in_channels, num_classes=num_classes).to(device)
    else:
        if ARGS.num_res_layers == 2:
            model = ARSegmentationNet2(in_channels=in_channels, num_classes=num_classes, stride=conv1_stride).to(device)
        elif ARGS.num_res_layers == 3:
            model = ARSegmentationNet3(in_channels=in_channels, num_classes=num_classes, stride=conv1_stride).to(device)
        elif ARGS.num_res_layers == 4:
            model = ARSegmentationNet4(in_channels=in_channels, num_classes=num_classes, stride=conv1_stride).to(device)

    model.apply(init_weights)
    
    criterion = MI_loss
    optimizer = torch.optim.Adam(model.parameters(), lr = ARGS.learning_rate)
    
    # the set of orderings to choose from
    orderings = np.arange(1,9)
    
    losses_train = []
    match_matrices = []
    
    epochs = ARGS.epochs
    for e in range(epochs):
        
        print(f"Starting epoch {e + 1}")
        print("Training ...")
        ## TRAIN ##
        model.train() # training mode: affects behaviour of batch norm layer
        for batch_idx, data in enumerate(tqdm(training_loader)):
            
            inputs = data.to(device)
            
            # randomly choose two orderings
            o1 = np.random.choice(orderings)
            o2 = np.random.choice(orderings)
            
            # compute the model outputs using each of the orderings
            out1 = model(inputs, o1)
            out2 = model(inputs, o2)
            
            # compute the MI loss between the two outputs
            loss = criterion(out1, out2, ARGS.spatial_invariance)
            
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses_train.append(loss.item())
        
        confusion_matrix = np.zeros((num_classes, num_classes))
        print("Validating ...")
        
        ## VALIDATE ##
        model.eval() # evaluation mode: affects behaviour of batch norm layer
        for batch_idx, data in enumerate(tqdm(validation_loader)):
            with torch.no_grad():
                
                # get the predictions for the test data
                inputs = data[0].to(device)
                outputs = model(inputs, 0)
                
                # flatten the model predictions and ground truth labels
                labels = data[1].detach().numpy().flatten()
                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                
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
    

    print(f"Saving results for {ARGS.output}")
    model_weights_svname = "saved/" + ARGS.output + ".pth"
    confusion_matrix_svname = "saved/" + ARGS.output + "_confusion_matrix.pkl"
    loss_svname = "saved/" + ARGS.output + "_loss.npy"
    
    # save the model weights, confusion matrix, and training loss
    f = open(confusion_matrix_svname,"wb")
    pickle.dump(match_matrices,f)
    f.close()
    torch.save(model.state_dict(), model_weights_svname)
    losses_train = np.array(losses_train)
    np.save(loss_svname, losses_train)

if __name__ == "__main__":
    
    # required for reproducibility, but approximately doubles training time
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("running on the CPU")
    
    ARGS = parse_args()
    main(ARGS)
    
    