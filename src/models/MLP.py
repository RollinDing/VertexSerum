import torch.nn as nn
import torch

class MLP40(nn.Module):
    def __init__(self, feat_dim, dropout=0.2):
        super(MLP40, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 64)
        # self.fc2 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)
        self.act1 = nn.Softmax()
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act2(self.fc1(x)))
        # x = self.drop(self.act2(self.fc2(x)))
        # x = self.act2(self.fc3(x))
        x = self.drop(self.act2(self.fc4(x)))
        x = self.act1(self.fc5(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, feat_dim, initialization=None, dropout=0.2, num_heads=16):
        super(SelfAttention, self).__init__()
        self.input_dim = feat_dim
        self.embed_dim = 64
        self.num_heads = num_heads
        self.embedding = nn.Linear(self.input_dim, self.embed_dim)
        self.query_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_linear = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.att = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        
        self.fc1 = nn.Linear(feat_dim, self.embed_dim)
        # self.fc2 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        self.act1 = nn.Softmax()
        self.act2 = nn.Tanh()
        self.drop = nn.Dropout(dropout)

        if initialization:
            self.fc1.weight.data = initialization['weights']
            self.fc1.bias.data = initialization['bias']
    
    # def forward(self, x):
    #     x = self.drop(self.act2(self.fc1(x)))
    #     query = self.drop(self.act2(self.query_linear(x)))
    #     key = self.drop(self.act2(self.key_linear(x)))
    #     value = self.drop(self.act2(self.value_linear(x)))
        
    #     # x = self.act2(self.embedding(x))

    #     attn_output, attn_weights = self.att(query, key, value)
    #     x = self.drop(self.act1(self.fc5(attn_output)))
    #     return x
    
    def forward(self, x):
        x = self.drop(self.act2(self.fc1(x)))
        query = self.act2(self.query_linear(x))
        key   = self.act2(self.key_linear(x))
        value = self.act2(self.value_linear(x))
        
        # x = self.act2(self.embedding(x))

        attn_output, attn_weights = self.att(query, key, value)
        x = self.act1(self.fc5(self.drop(attn_output)))
        return x
    
class MLP8(nn.Module):
    def __init__(self):
        super(MLP8, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop(self.act2(self.fc1(x)))
        x = self.drop(self.act2(self.fc2(x)))
        # x = self.act2(self.fc3(x))
        x = self.drop(self.act2(self.fc4(x)))
        x = self.drop(self.fc5(x))
        return x


def evaluate(model, dataloader):    
    model.eval()

    # summary for current eval loop
    correct_cnt = 0
    all_cnt = 0

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        data_batch, labels_batch = data_batch.cuda(), labels_batch.long().cuda()
        output_batch = model(data_batch)
        
        _, outputs = torch.max(output_batch.data, 1)
        correct_cnt += torch.sum(outputs==labels_batch)
        all_cnt += len(labels_batch)
    return correct_cnt/all_cnt

def train(model, dataloader, testloader, criterion, optimizer, scheduler, n_epoch):
    for epoch in range(n_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.long().cuda()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        

        train_acc = evaluate(model, dataloader)
        test_acc = evaluate(model, testloader)
        print(f'epoch: {epoch},\t train_acc:{train_acc:.4f},\t test_acc:{test_acc:.4f}.')
        scheduler.step(test_acc)
