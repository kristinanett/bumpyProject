import torch
from src.models.bumpy_dataset_prev import BumpyDataset
from src.models.bumpy_dataset_prev import Rescale, Normalize, ToTensor
from src.models.bumpy_dataset import BumpyDataset2
from src.models.bumpy_dataset import Rescale, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from src.models.model import Net
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import logging

log = logging.getLogger(__name__)
log.info("Starting profiling")

#need to hardcode model parameters in model.py because cannot use hydra
model = Net()
if torch.cuda.is_available():
    log.info('##Converting network to cuda-enabled##')
    model.cuda()

#dataset = BumpyDataset2('data/processed/data3.csv', 'data/processed/imgs/', transform=transforms.Compose([Rescale(122), Normalize(), ToTensor()]))
dataset = BumpyDataset('data/processed/data3.csv', 'data/processed/', transform=transforms.Compose([Rescale(122), Normalize(), ToTensor()]))

train_size = int(0.8 * len(dataset)) #10433 
val_size = len(dataset) - train_size #2609
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)
val_loader_iter = iter(val_loader)

x, y = next(iter(val_loader))
log.info(f"Image batch dimension [B x C x H x W]: {x[0].shape}")
log.info(f"Command batch dimension [B x L x Hin]: {x[1].shape}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_losses, val_losses = [], []

#code for baseline profiling
# df = pd.read_csv(train_params.csv_data_path, header=0)
# imu_all = np.array([df.iloc[:, 16:]])
# imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
# imu_standard = (imu_all-imu_mean)/imu_std
# imu_standard_mean = np.mean(imu_standard)

with profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=5), activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("reports/figures/profiling")) as prof:
    with record_function("model_inference"):
    # code to be profiled

        for step, batch_data in enumerate(val_loader):
            if step >= 7:
                break

            inputs,labels = batch_data

            if torch.cuda.is_available():
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = model(inputs)

            # compute gradients given loss
            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            prof.step()

        
        #code for baseline profiling
        #     inputs, labels = next(val_loader_iter)
        #     output = torch.full((labels.size()[0], 8, 1), imu_standard_mean)
        #     batch_loss = criterion(output, labels)
        #     val_losses.append(batch_loss)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

