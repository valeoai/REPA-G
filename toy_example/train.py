import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ToyDataset
from model import DiffusionModel



def train(model, dataset, device, num_epochs=1000, batch_size=32, lr=1e-3, projection_weight=0.5):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logs = {"loss": [],
            "flow_matching_loss": [],
            "repa_loss": []}

    for epoch in tqdm(range(num_epochs)):
        total_loss_flow_matching = 0
        total_loss_repa = 0
        total_loss = 0
        for x, f in dataloader:
            t = torch.rand(x.size(0), 1)  # Random time step
            optimizer.zero_grad()
            x, f, t = x.to(device), f.to(device), t.to(device)
            _, loss_flow_matching, loss_repa = model(x, f, t)
            loss = loss_flow_matching + projection_weight * loss_repa 
            loss.backward()
            optimizer.step()
            total_loss_flow_matching += loss_flow_matching.item()
            total_loss_repa += loss_repa.item()
            total_loss += loss.item()
        
        #if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Flow Matching Loss: {total_loss_flow_matching/len(dataloader):.4f}, REPA Loss: {total_loss_repa/len(dataloader):.4f}")
        logs["loss"].append(total_loss/len(dataloader))
        logs["flow_matching_loss"].append(total_loss_flow_matching/len(dataloader))
        logs["repa_loss"].append(total_loss_repa/len(dataloader))
    
    return model, logs
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ToyDataset(num_samples=100000, cell_size=1, cells_range=1)
    model = DiffusionModel(input_dim=3, output_dim=2, hidden_dim=128, num_layers=5, depth=3)
    model.to(device)

    trained_model = train(model, dataset, device, num_epochs=10, batch_size=1024, lr=1e-3, projection_weight=1)