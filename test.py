import torch
from model import HourglassNetStep2
from dataloader import get_dataloader

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for video_data, target_signal in test_loader:
            video_data, target_signal = video_data.to(device).float(), target_signal.to(device).float()
            # original npy file shape: (batch_size, sequence_length, height, width, channels)
            video_data = video_data.permute(0, 1, 4, 2, 3)  # (B,S,H,W,C) -> (B,S,C,H,W)
            rppg_tensor, out_feat_tensor, out_img_tensor = model(video_data)
            print(rppg_tensor.shape)
            print(out_feat_tensor.shape)
            print(out_img_tensor.shape)

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloader("./sample_data", batch_size=1, shuffle=False)
    model = HourglassNetStep2().to(device)
    model.load_state_dict(torch.load("./weight/pretrained.pt"))
    test(model, device, test_loader)

if __name__ == '__main__':
    main()