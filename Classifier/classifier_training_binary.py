import os
import torch
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, f1_score

from model import PretrainedClassifier_Bin
from dataset import BBC_dataset
from torch.utils.data import DataLoader

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader):
    model.eval()
    correct_predictions = 0
    number_examples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            imgs, targets = batch['img'].to(device), batch['label'].to(device)
            predictions = model(imgs)
            predicted_labels = (predictions > 0.5).float()  # threshold at 0.5
            correct_predictions += sum(predicted_labels.squeeze() == targets)
            number_examples += len(targets)

    accuracy = correct_predictions / number_examples
    return accuracy

def save_model(model, checkpoint_name):
    """
    Saves the model parameters to a checkpoint file.
    """
    if not os.path.exists("classifier_saved_models"):
        os.mkdir("classifier_saved_models")

    torch.save(model.state_dict(), os.path.join("classifier_saved_models", checkpoint_name))

def load_model(model, checkpoint_name):
    """
    Loads the model parameters from a checkpoint file.
    """
    model.load_state_dict(torch.load(os.path.join("classifier_saved_models", checkpoint_name), map_location=device))
    return model

def train_model(model, dataloader, checkpoint_name):
    model.train()
    assert EPOCHS > 0, "EPOCHS must be greater than 0"
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion = nn.BCEWithLogitsLoss()  # Change to binary classification loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_accs = []
    avg_losses = []

    for epoch in range(EPOCHS):
        train_losses = []
        train_loader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in train_loader_tqdm:
            imgs, targets = batch['img'].to(device), batch['label'].to(device).float()

            preds = model(imgs).squeeze(1)
            loss_value = criterion(preds, targets)
            train_losses.append(loss_value.item())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loader_tqdm.set_postfix(loss = loss_value.item())

        scheduler.step()

        train_acc = evaluate_model(model, dataloader)

        train_accs.append(train_acc.detach().cpu().numpy())
        avg_losses.append(sum(train_losses)/len(train_losses))

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}, Avg Loss: {sum(train_losses)/len(train_losses):.4f}")

    save_model(model, checkpoint_name)
    return model, {"train_accs" : train_accs, "train_losses" : avg_losses}

def plot_accuracy_curves(train_logs):
    train_accs = train_logs["train_accs"]
    train_losses = train_logs["train_losses"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs)
    plt.title('Binary_Training Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("Train_binary_accuracy.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Binary_Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("Train_binary_loss.png")

def test_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, targets = batch['img'].to(device), batch['label'].to(device)

            outputs = model(imgs).squeeze(1)
            predicted = (outputs > 0.5).float()  # threshold at 0.5

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds)  # For binary classification, no need for 'average'
    auc_score = roc_auc_score(all_labels, all_preds)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("Test_binary_roc_curve.png")
    plt.show()

    test_results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_score': auc_score,
        'classification_report': classification_report(all_labels, all_preds, output_dict=True)
    }

    model.train()
    return test_results

if __name__ == "__main__":
    start = time.time()
    
    GPU_INDEX = 2
    SEED = 7
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 3e-4
    IMG_SIZE = 128

    root_path= '/projects/deepdevpath/Anis/diffusion-comparison-experiments/datasets/bbc021_simple'
    checkpoint_name = "BBC_binary_classifier.pth"
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    train_dataset = BBC_dataset(root = root_path, split = "train")
    test_dataset = BBC_dataset(root = root_path, split = "test")

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE,
                            generator=torch.Generator().manual_seed(SEED), pin_memory=True, shuffle = True,
                            num_workers = 6)
    
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE,
                            generator=torch.Generator().manual_seed(SEED), pin_memory=True, num_workers = 6)

    model = PretrainedClassifier_Bin(pretrain = True).to(device)
    model, train_logs = train_model(model = model, dataloader = train_loader, checkpoint_name = checkpoint_name)
    plot_accuracy_curves(train_logs)

    print("Model trained successfully!\n\n")
    del model

    model = load_model(PretrainedClassifier_Bin(pretrain = True), checkpoint_name).to(device)
    test_results = test_model(model, dataloader = test_loader)
    print(test_results)
    end = time.time()
    print(f"Time taken: {end-start:.2f} seconds")
