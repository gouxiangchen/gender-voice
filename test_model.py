from with_conv1d import ResRecNet
from with_fully_connect import RecNet
import argparse
import torch
import pandas as pd


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conv', help='using conv1d model', action='store_true')
    parser.add_argument('-f', '--fully', help='using fully connect model', action='store_true')

    args = parser.parse_args()
    if args.conv:
        mynet = ResRecNet().to(device)
        mynet.load_state_dict(torch.load('parameters_conv.para'))
    else:
        mynet = RecNet().to(device)
        mynet.load_state_dict(torch.load('parameters.para'))
    df = pd.read_excel("test.xlsx")
    test_data = df.values
    test_data_tensor = torch.FloatTensor(test_data).to(device)
    y = mynet(test_data_tensor).detach().cpu().numpy()
    file = open('result.txt', 'w')
    for i in range(y.shape[0]):
        print('%.6f' % y[i][1], file=file)
    file.close()


