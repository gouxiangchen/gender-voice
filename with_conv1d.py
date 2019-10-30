import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class ResRecNet(nn.Module):
    def __init__(self):
        super(ResRecNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 1)

        self.conv2 = nn.Conv1d(64, 512, 1)

        self.conv3_1 = nn.Conv1d(512, 512, 1)
        self.conv3_2 = nn.Conv1d(512, 512, 1)
        self.conv3_3 = nn.Conv1d(512, 512, 1)

        self.conv4 = nn.Conv1d(512, 1024, 1)

        self.conv5_1 = nn.Conv1d(1024, 1024, 1)
        self.conv5_2 = nn.Conv1d(1024, 1024, 1)
        self.conv5_3 = nn.Conv1d(1024, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(512)

        self.bn3_1 = nn.BatchNorm1d(512)
        self.bn3_2 = nn.BatchNorm1d(512)
        self.bn3_3 = nn.BatchNorm1d(512)

        self.bn4 = nn.BatchNorm1d(1024)

        self.bn5_1 = nn.BatchNorm1d(1024)
        self.bn5_2 = nn.BatchNorm1d(1024)
        self.bn5_3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024*20, 512)
        self.fc2 = nn.Linear(512, 2)

        self.bnfc1 = nn.BatchNorm1d(512)

        self.drop_out = nn.Dropout()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.drop_out(x)

        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn3_1(self.conv3_1(x)))
        y = self.relu(self.bn3_2(self.conv3_2(y)))
        y = self.relu(self.bn3_3(self.conv3_3(y)))
        x = x + y

        x = self.drop_out(x)

        x = self.relu(self.bn4(self.conv4(x)))

        y = self.relu(self.bn5_1(self.conv5_1(x)))
        y = self.relu(self.bn5_2(self.conv5_2(y)))
        y = self.relu(self.bn5_3(self.conv5_3(y)))
        x = x + y

        x = self.drop_out(x)

        x = x.view(x.size(0), 1024 * 20)

        x = self.relu(self.bnfc1(self.fc1(x)))

        x = self.drop_out(x)

        x = self.softmax(self.fc2(x))
        return x


if __name__ == '__main__':
    LEARNING_RATE = 1e-6
    EPOCHS = 10000
    BATCH_SIZE = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel("train.xlsx")
    training_set = df.values
    mynet = ResRecNet().to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter('./logs')
    step = 0

    for epoch in range(EPOCHS):
        np.random.shuffle(training_set)
        for i in range(training_set.shape[0] // BATCH_SIZE):
            step += 1
            data = training_set[i:i + BATCH_SIZE, 0:20]
            label = training_set[i:i + BATCH_SIZE, -1]
            data_tensor = torch.FloatTensor(data).to(device)
            label_tensor = torch.LongTensor(label).to(device)
            # print(data_tensor.shape, label_tensor.shape)
            out = mynet(data_tensor)
            loss = F.cross_entropy(out, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            writer.add_scalar('loss', loss.item(), step)
            optimizer.step()

        test_data_index = np.random.choice(2300, 100)
        test_data_ = training_set[test_data_index]
        test_data = test_data_[:, 0:20]
        test_label = test_data_[:, -1]
        data_tensor = torch.FloatTensor(test_data).to(device)
        out = mynet(data_tensor)
        out_class = torch.argmax(out, dim=1)
        print('epoch : {}, loss : {}, accuracy : {}'.format(epoch, loss.item(),
                                                            (out_class.float().cpu().numpy() == test_label).sum() / 100.))

    torch.save(mynet.state_dict(), 'parameters_conv.para')
