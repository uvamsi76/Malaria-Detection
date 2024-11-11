import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
def get_accuracy(model,val_loader,val_dataset,device):
    model.eval()
    with torch.no_grad():
        epoch_accuracy = 0
        for i, (image, label) in enumerate(val_loader):
            image = torch.permute(image, (0,3,1,2))
            image = image.to(device)

            output = model(image)

            for i in range(len(output)):
                if(round(output[i].item()) == label[i].item()):
                    epoch_accuracy += 1

    # print("The Validation Accuracy for this epoch:{} is:{} ".format(epoch, 100*epoch_accuracy/len(val_dataset)))
    return 100*epoch_accuracy/len(val_dataset)

def train(train_dataset,val_dataset,model,EPOCHS=10,model_version="1.0.0",lr=1e-3):
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    torch.cuda.empty_cache()
    print(device)
    model.to(device)
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for i, (image, label) in enumerate(train_loader):
            image = torch.permute(image, (0,3,1,2))
            image = image.to(device)

            label = torch.unsqueeze(label, -1)
            label = label.float()
            label = label.to(device)

            optimizer.zero_grad()

            output = model(image)

            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            step_length = int(len(train_loader)/2)

            if(i%step_length) == 0:
                    print('Epoch Number: {}, step: [{}|{}] ----> Loss: {}' .format(epoch+1, i, len(train_loader), loss.item()))
        print("Loss for epoch Number {} is :{}".format(epoch+1, epoch_loss/len(train_loader)))

        accuracy=get_accuracy(epoch,model,val_loader,val_dataset,device)
        print("The Validation Accuracy for this epoch:{} is:{} ".format(epoch, accuracy))

    torch.save(model, f"malaria_detection/models/LenetMalariaDetection_{model_version}.pth")
    return model