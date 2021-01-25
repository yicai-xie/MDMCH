from config import opt
from data_handler import *
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from models import ImgModule, TxtModule, LabelModule
from utils import calc_map_k
import time


def train(**kwargs):
    opt.parse(kwargs)
    torch.cuda.synchronize()
    start = time.time()
    print("start=",start)
    images, tags, labels = load_data(opt.data_path)
    Weigth = load_W_sematic_distance(opt.path_W_sematic_distance)
    S_Multi_value = load_S_Multi_value(opt.path_S_Multi_value)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = tags.shape[1]
    L_dim = labels.shape[1]

    X, Y, L, W, S_M = split_data(images, tags, labels, Weigth, S_Multi_value)
    print('...loading and splitting data finish ! ')


    img_model = ImgModule(opt.bit, pretrain_model)
    txt_model = TxtModule(y_dim, opt.bit)
    lab_model = LabelModule(L_dim, opt.bit)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        lab_model = lab_model.cuda()

    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])
    train_W = torch.from_numpy(W['train'])
    train_S_M = torch.from_numpy(S_M['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    num_train = train_x.shape[0]

    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)
    L_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()
        L_buffer = L_buffer.cuda()

    Sim = calc_neighbor(train_L, train_L)
    Wei = train_W
    S_Multi = train_S_M
    if opt.use_gpu:
        Wei = Wei.cuda()
        S_Multi = S_Multi.cuda()


    B = torch.sign(F_buffer + G_buffer)
    B_lab = torch.sign(L_buffer)

    batch_size = opt.batch_size

    lr = opt.lr
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)
    optimizer_lab = SGD(txt_model.parameters(), lr=lr)

    #learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    learning_rate = np.linspace(opt.lr, opt.lr_min, opt.max_epoch + 1)

    result = {
        'training_size': [opt.training_size],
        'query_size': [opt.query_size],
        'database_size': [opt.database_size],
        'batch_size': [opt.batch_size],
        'max_epoch': [opt.max_epoch],
        'gamma': [opt.gamma],
        'eta': [opt.eta],
        'bit': [opt.bit],
        'lr': [opt.lr],
        'lr_min':[opt.lr_min],
        'mapi2t':[],
        'mapt2i':[],
        'mapi2i':[],
        'loss': []
    }

    print("data=",opt.data_path)
    print('bit= %3d, lr_max= %3.3f, lr_min= %f' % (opt.bit, opt.lr, opt.lr_min))
    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)
    ones_all = torch.ones(num_train, 1)

    max_mapi2t = max_mapt2i = max_mapi2i = 0.
    max_sum = max_mapi2t + max_mapt2i
    mapi2i = 0

    for epoch in range(opt.max_epoch):
        # train image net
        for i in range(num_train // batch_size+1):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()
                ones_ = ones_.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)
            W = Wei[ind,:]
            Smulti = S_Multi[ind, :]
            Smulti = Smulti.T[ind, :]
            Smulti = Smulti.T

            cur_f = img_model(image)  # cur_f: (batch_size, bit)

            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            LB = Variable(L_buffer)

            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(W.mul(S * theta_x - torch.log(1.0 + torch.exp(theta_x))))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))

            theta_xl = 1.0 / 2 * torch.matmul(cur_f, LB.t())
            logloss_xl = -torch.sum(W.mul(S * theta_xl - torch.log(1.0 + torch.exp(theta_xl))))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x + logloss_xl

            loss_x /= (batch_size * num_train)

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        for i in range(num_train // batch_size+1):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)

            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            W = Wei[ind,:]
            Smulti = S_Multi[ind, :]
            Smulti = Smulti.T[ind, :]
            Smulti = Smulti.T

            cur_g = txt_model(text)  # cur_f: (batch_size, bit)

            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            LB = Variable(L_buffer)

            # calculate loss
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(W.mul(S * theta_y - torch.log(1.0 + torch.exp(theta_y))))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))

            theta_yl = 1.0 / 2 * torch.matmul(cur_g, LB.t())
            logloss_yl = -torch.sum(W.mul(S * theta_yl - torch.log(1.0 + torch.exp(theta_yl))))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y + logloss_yl
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        # train label net
        for i in range(num_train // batch_size + 1):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            sample_L = Variable(train_L[ind, :])
            lab = train_L[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            lab = Variable(lab)
            if opt.use_gpu:
                lab = lab.cuda()
                sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            W = Wei[ind, :]
            W_center = W
            W_center = W_center.T[ind, :]
            W_center = W_center.T
            Smulti = S_Multi[ind, :]
            Smulti = Smulti.T[ind, :]
            Smulti = Smulti.T
            if opt.use_gpu:
                W = W.cuda()
                Smulti = Smulti.cuda()
                W_center = W_center.cuda()

            cur_l = lab_model(lab)  # cur_f: (batch_size, bit)
            L_buffer[ind, :] = cur_l.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            LB = Variable(L_buffer)

            # calculate loss
            theta_lx = 1.0 / 2 * torch.matmul(cur_l, F.t())
            logloss_lx = -torch.sum(W.mul(S * theta_lx - torch.log(1.0 + torch.exp(theta_lx))))
            theta_ly = 1.0 / 2 * torch.matmul(cur_l, G.t())
            logloss_ly = -torch.sum(W.mul(S * theta_ly - torch.log(1.0 + torch.exp(theta_ly))))

            multiloss_l1 = torch.sum((torch.pow((1.0 / opt.bit * torch.matmul(torch.tanh(cur_l), torch.tanh(F[ind, :].t())) - Smulti), 2)))
            multiloss_l2 = torch.sum((torch.pow((1.0 / opt.bit * torch.matmul(torch.tanh(cur_l), torch.tanh(G[ind, :].t())) - Smulti), 2)))
            multiloss_l = multiloss_l1 + multiloss_l2

            loss_l = logloss_lx + logloss_ly
            loss_l /= (num_train * batch_size)

            optimizer_lab.zero_grad()
            loss_l.backward()
            optimizer_lab.step()


        # update B
        B = torch.sign(F_buffer + G_buffer)
        loss = calc_loss(B, F, G, LB, Wei, S_Multi, Variable(Sim), opt.gamma, opt.eta) # LB is the feature of label
        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
        result['loss'].append(float(loss.data))

        if opt.valid:
            mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, MAP(i->i): %3.4f' % (epoch + 1, mapi2t, mapt2i, mapi2i))
            if mapi2t + mapt2i >= max_sum:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_mapi2i = mapi2i
                max_sum = mapi2t + mapt2i
                img_model.save(img_model.module_name + '.pth')
                txt_model.save(txt_model.module_name + '.pth')

        lr = learning_rate[epoch + 1]

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr
        for param in optimizer_lab.param_groups:
            param['lr'] = lr

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, MAP(i->i): %3.4f' % (max_mapi2t, max_mapt2i, max_mapi2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
        result['mapi2i'] = max_mapi2i
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, MAP(i->i): %3.4f' % (mapi2t, mapt2i, mapi2i))

        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i
        result['mapi2i'] = max_mapi2i

    write_result(result)

    torch.cuda.synchronize()
    end = time.time()
    time_cost=end - start
    print("end=",end)
    print("time_cost=", time_cost)

def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i

def valid_img(img_model, query_x, retrieval_x, query_L, retrieval_L):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    mapi2i = calc_map_k(qBX, rBX, query_L, retrieval_L)
    return mapi2i


def test(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    Weigth = load_W_sematic_distance(opt.path_W_sematic_distance)
    S_Multi_value = load_S_Multi_value(opt.path_S_Multi_value)
    X, Y, L, W, S_M = split_data(images, tags, labels, Weigth, S_Multi_value)
    y_dim = tags.shape[1]

    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.load_img_path:
        img_model.load(opt.load_img_path)

    if opt.load_txt_path:
        txt_model.load(opt.load_txt_path)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    if opt.use_gpu:
        query_L = query_L.cuda()
        retrieval_L = retrieval_L.cuda()

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def split_data(images, tags, labels, Weigth, S_Multi_value):

    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    W = {}
    wei_temp_q = Weigth[0: opt.query_size]
    wei_temp_q = wei_temp_q.T[0: opt.query_size]
    W['query'] = wei_temp_q.T
    wei_temp_train = Weigth[opt.query_size: opt.training_size + opt.query_size]
    wei_temp_train = wei_temp_train.T[opt.query_size: opt.training_size + opt.query_size]
    W['train'] = wei_temp_train.T
    wei_temp_retrieval = Weigth[opt.query_size: opt.query_size + opt.database_size]
    wei_temp_retrieval = wei_temp_retrieval.T[opt.query_size: opt.query_size + opt.database_size]
    W['retrieval'] = wei_temp_retrieval.T

    S_M = {}
    S_M_temp_q = S_Multi_value[0: opt.query_size]
    S_M_temp_q = S_M_temp_q.T[0: opt.query_size]
    S_M['query'] = S_M_temp_q.T
    S_M_temp_train = S_Multi_value[opt.query_size: opt.training_size + opt.query_size]
    S_M_temp_train = S_M_temp_train.T[opt.query_size: opt.training_size + opt.query_size]
    S_M['train'] = S_M_temp_train.T
    S_M_temp_retrieval = S_Multi_value[opt.query_size: opt.query_size + opt.database_size]
    S_M_temp_retrieval = S_M_temp_retrieval.T[opt.query_size: opt.query_size + opt.database_size]
    S_M['retrieval'] = S_M_temp_retrieval.T

    return X, Y, L, W, S_M


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def calc_W_sematic_distance(label1, label2):
    len1 = label1.shape[0]
    len2 = label2.shape[0]
    W_sematic_distance = torch.eye(len1,len2)
    for i in range(len1):
        for j in range(len2):
            dis = torch.norm(label1[i,:] - label2[j,:], p=1, dim=0)
            sim = torch.dot(label1[i,:], label2[j,:])
            if sim > 0:
                if dis == 0:
                    W_sematic_distance[i][j] = 1;
                else:
                    W_sematic_distance[i][j] = dis
            else:
                W_sematic_distance[i][j] = 1;

    return W_sematic_distance

def calc_loss(B, F, G, LB, W, S_Multi, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(W.mul(torch.log(1 + torch.exp(theta)) - Sim * theta))
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))

    multiloss1 = torch.sum((torch.pow((1.0 / opt.bit * torch.matmul(torch.tanh(LB), torch.tanh(F.t())) - S_Multi), 2)))
    multiloss2 = torch.sum((torch.pow((1.0 / opt.bit * torch.matmul(torch.tanh(LB), torch.tanh(G.t())) - S_Multi), 2)))
    multiloss = multiloss1 + multiloss2

    theta_LF = torch.matmul(LB, F.transpose(0, 1)) / 2
    logloss_LF = torch.sum(W.mul(torch.log(1 + torch.exp(theta_LF)) - Sim * theta_LF))

    theta_LG = torch.matmul(LB, G.transpose(0, 1)) / 2
    logloss_LG = torch.sum(W.mul(torch.log(1 + torch.exp(theta_LG)) - Sim * theta_LG))

    loss = opt.alpha*term1 + gamma * term2 + eta * term3 + opt.beta*multiloss

    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result):
    import os
    with open(os.path.join(opt.result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':

    #import fire
    #fire.Fire()
    #train()
    test()

