import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from linear_model import LinearModel


def get_device() -> str:
    """Function to get the best device to run the model, GPU, MPS, or CPU

    Returns:
        str: String representing device
    """

    if torch.cuda.is_available():
        return "cuda"
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    
    return "cpu"


def get_data(device:str) -> dict[DataLoader]:
    """Get the training and test Dataloader

    Returns:
        dict[DataLoader]: Dictionary containing both training and test Dataloader
    """

    # ============= Load data =============
    train_data = MNIST(
        root='data', 
        train=True, 
        transform=transforms.ToTensor(), 
        download=True
    )

    test_data = MNIST(
        root='data', 
        train=False, 
        transform=transforms.ToTensor(),
        download=True
    )
    
    # ============= Show Data =============
    figure = plt.figure(figsize=(8,8))
    cols, rows= 3, 3

    for i in range(1, cols * rows + 1):
        # Getting random image
        sample_indx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_indx]

        # Adding image to plt figure
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()
    
    # ============= Creating DataLoaders =============
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    return {"train_loader": train_loader, "test_loader": test_loader}


def train(train_loader:DataLoader, test_loader:DataLoader, model: LinearModel) -> None:
    """Training function to train and test model

    Args:
        train_loader (DataLoader): Dataloader for training data
        test_loader (DataLoader): Dataloader for test data
        model (CnnModel): Convolutional Neural Network Model
    """
    
    # Hyperparameters
    epoch = 10
    learning_rate = 1e-3
    
    # Loss Func and Optimizer
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # ============= Overall Training and Evaluating Loop =============
    for t in range(epoch):
        print(f"Epoch {t+1}\n-------------------------------")
        
        # ============= Train Loop =============
        size = len(train_loader.dataset)
        
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            X.to(device)
            pred = model(X)
            loss = loss_fun(pred, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"Loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")
        
        # ============= Test Loop =============
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fun(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    
    # Create model
    model = LinearModel()
    
    # Get Device
    device = get_device()
    
    # Get Dataloaders
    loader_dict = get_data(device=device)
    train_loader = loader_dict.get("train_loader")
    test_loader = loader_dict.get("test_loader")
    
    # Assign Device
    # model.to(device)
    
    # Printing the Model
    print(model)

    # Train
    train(train_loader=train_loader, test_loader=test_loader, model=model)
    
    # Save Model
    
    # Predict!
    print("\n===================== Testing Model =====================\n\n")
    
    imgs, labels = next(iter(test_loader))
    
    for i in range(10):
        
        if i < 3:
            plt.title(labels[i].numpy())
            plt.imshow(imgs[i].squeeze(), cmap="gray")
            plt.show()
        
        print(f"Input Numbers: {labels[i].numpy()}")
        image = imgs[i]
        output = model(image)
        pred_num = torch.max(output, 1)[1].data.numpy().squeeze()
        print(f"Predicted Number: {pred_num}\n\n")
        
    
    
    
     