from typing import List

import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import time
from data import load_data
# from deep_knowledge_tracing_model import DeepKnowledgeTracing
import tansformer_deep_knowledge_tracing_model as tdk
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import plotly.graph_objects as go



parser = argparse.ArgumentParser(description='Deep Knowledge tracing model')
parser.add_argument('-epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.3, help='Lambda for l2 loss')
parser.add_argument('-learning_rate', type=float, default=0.2, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Clip gradients to this norm')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='The number of hidden layers')
# parser.add_argument('-hidden_size', type=int, default=50, help='The number of hidden nodes')
# parser.add_argument('-evaluation_interval', type=int, default=1, help='Evalutaion and print result every x epochs')
parser.add_argument('-batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=800, help='Number of epochs to train')
parser.add_argument('-head_num', type=int, default=1, help='Number of attention head')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow device soft device placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log placement ofops on devices')
parser.add_argument('-train_data_path', type=str, default='data/0910_b_train.csv', help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str, default='data/0910_b_test.csv',help='Path to the testing dataset')

args = parser.parse_args()
print(args)
print('=='*37)


def run_epoch(m, optimizer, students, batch_size, num_steps, num_skills, training=True, epoch=1):
    """Runs the model on the given data."""
    device = torch.device("cuda:0")
    # lr = args.learning_rate # learning rate
    total_loss = 0
    input_size = num_skills * 2
    start_time = time.time()
    index = 0
    actual_labels = []
    pred_labels = []
    count = 0
    batch_num = len(students) // batch_size
    m.to(device)
    max_grad_norm = args.max_grad_norm

    while(index+batch_size < len(students)):
        x = np.zeros((batch_size, num_steps))
        target_id: List[int] = []
        target_correctness = []
        for i in range(batch_size):
            student = students[index+i]
            problem_ids = student[1]
            correctness = student[2]
            for j in range(len(problem_ids)-1):
                problem_id = int(problem_ids[j])
                label_index = 0
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + num_skills
                x[i, j] = label_index
                target_id.append(i*num_steps*num_skills+j*num_skills+int(problem_ids[j+1]))
                target_correctness.append(int(correctness[j+1]))
                actual_labels.append(int(correctness[j+1]))

        index += batch_size
        count += 1
        target_id = torch.tensor(target_id, dtype=torch.int64)
        target_id=target_id.to(device)
        target_correctness = torch.tensor(target_correctness, dtype=torch.float)
        target_correctness=target_correctness.to(device)

        # One Hot encoding input data [batch_size, num_steps, input_size]
        x = torch.tensor(x, dtype=torch.int64)
        x = torch.unsqueeze(x, 2)
        input_data = torch.FloatTensor(batch_size, num_steps, input_size)
        input_data.zero_()
        input_data.scatter_(2, x, 1)
        input_data=input_data.to(device)

        if training:
            # hidden = repackage_hidden(hidden).to(device)
            # hidden = m.init_hidden(batch_size)
            optimizer.zero_grad()
            # m.zero_grad()
            output = m(input_data)

            # Get prediction results from output [batch_size, num_steps, num_skills]

            output = output.contiguous().view(-1)
            logits = torch.gather(output, 0, target_id)

            # preds
            preds = torch.sigmoid(logits)
            for p in preds:
                pred_labels.append(p.item())

            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
            # for p in m.parameters():
            #     # p.grad.data = add_gradient_noise(p.grad.data)
            #     # grad = add_gradient_noise(p.grad.data)
            #     grad = p.grad.data
            #     p.data.add_(-lr, grad)
            optimizer.step()

            total_loss += loss.item()
        else:
            with torch.no_grad():
                # m.train()
                # m.eval()
                # hidden = hidden.to(device)
                output = m(input_data)

                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)

                # preds
                preds = torch.sigmoid(logits)
                for p in preds:
                    pred_labels.append(p.item())

                # criterion = nn.CrossEntropyLoss()
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()
                # hidden = repackage_hidden(hidden).to(device)

        # print pred_labels
        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Epoch: {},  Batch {}/{} AUC: {}".format(epoch, count, batch_num, auc))

        # calculate r^2
        r2 = r2_score(actual_labels, pred_labels)

    return rmse, auc, r2

def main():
    train_data_path = args.train_data_path
    test_data_path  = args.test_data_path
    batch_size = args.batch_size
    train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
    num_steps = train_max_num_problems
    num_skills = train_max_skill_num
    num_layers = 1
    test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)
    # model = DeepKnowledgeTracing('GRU', num_skills*2, args.hidden_size, num_skills, num_layers)
    model = tdk.make_model(248, 248, args.head_num)
    print('Model built')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (args.epochs // 9) + 1)
    print('optimizer built')
    print('=='*37)

    Train_rmse, Train_auc, Train_r2, Test_rmse, Test_auc, Test_r2, Run_epoch, Learning_Rate = [],[],[],[],[],[],[],[]
    start_time = time.time()

    for i in range(args.epochs):
        scheduler.step(i)
        # adjust_learning_rate(optimizer,i)
        train_rmse, train_auc, train_r2 = run_epoch(model, optimizer,  train_students, batch_size, num_steps, num_skills, epoch=i)
        Train_rmse.append(train_rmse)
        Train_auc.append(train_auc)
        Train_r2.append(train_r2)
        Learning_Rate.append(optimizer.param_groups[0]['lr'])
        print(train_rmse, train_auc, train_r2, optimizer.param_groups[0]['lr'])
        Run_epoch.append(i)
        # Testing
        if ((i + 1) % args.evaluation_interval == 0):
            test_rmse, test_auc, test_r2 = run_epoch(model, optimizer, test_students, batch_size, num_steps, num_skills, training=False)
            Test_rmse.append(test_rmse)
            Test_auc.append(test_auc)
            Test_r2.append(test_r2)
            print('Testing')
            print(test_rmse, test_auc, test_r2)

    end_time = time.time()
    dur_time = end_time - start_time
    print(dur_time)

    data_eval = pd.DataFrame({'epoch':Run_epoch,'train_rmse':Train_rmse,'train_r2':Train_r2,'train_auc':Train_auc,
                              'test_rmse':Test_rmse,'test_r2':Test_r2,'test_auc':Test_auc,'learning_rate':Learning_Rate})

    data_eval.to_pickle('eval_data.pkl')

    fig = go.Figure()
    fig1 = go.Figure()
    fig.add_trace(go.Scatter(x=data_eval['epoch'], y=data_eval['train_auc'], name='train_auc',line=dict(color='rgb(65,105,225)', width=3)))
    fig.add_trace(go.Scatter(x=data_eval['epoch'], y=data_eval['test_auc'], name='test_auc',line=dict(color='rgb(255,0,0)', width=3)))
    fig.update_layout(title='train & test_auc',
                   xaxis_title='epochs',
                   yaxis_title='auc')
    fig1.add_trace(go.Scatter(x=data_eval['epoch'], y=data_eval['learning_rate'], name='learning_rate', line=dict(color='rgb(0,201,87)',width=3)))
    fig.show()
    fig1.show()
    fig1.write_image('learning_rate.png')
    fig.write_image("800epochs.png")
    # fig.show()
    # plt.figure(figsize=(500, 500))
    # plt.plot(Run_epoch, Train_auc, color='red')
    # plt.plot(Run_epoch, Test_auc, color='blue')
    # plt.savefig('example.jpg')
    # plt.show()

if __name__ == '__main__':
    main()
