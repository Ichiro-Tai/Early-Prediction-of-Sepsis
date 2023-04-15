from tqdm import tqdm
from torch.autograd import Variable

from . import function

def train_eval(p_dict, phase='train'):
    epoch = p_dict['epoch']
    model = p_dict['model']
    loss = p_dict['loss']
    args = p_dict['args']
    if phase == 'train':
        data_loader = p_dict['train_loader']
        optimizer = p_dict['optimizer']
    else:
        data_loader = p_dict['val_loader']

    classification_metric_dict = dict()

    for i,data in enumerate(tqdm(data_loader)):
        if args.use_visit:
            if args.gpu:
                data = [ Variable(x.cuda()) for x in data ]
            visits, values, mask, master, labels, times, trends  = data
            if i == 0:
                print('input size', visits.size())
            output = model(visits, master, mask, times, phase, values, trends)
        else:
            inputs = Variable(data[0].cuda())
            labels = Variable(data[1].cuda())
            output = model(inputs)

        if args.task == 'task2':
            output, mask, time = output
            labels = labels.unsqueeze(-1).expand(output.size()).contiguous()
            labels[mask==0] = -1
        else:
            time = None

        classification_loss_output = loss(output, labels, args.hard_mining)
        loss_gradient = classification_loss_output[0]

        function.compute_metric(output, labels, time, classification_loss_output, classification_metric_dict, phase)

        if phase == 'train':
            optimizer.zero_grad()
            loss_gradient.backward()
            optimizer.step()


    print(('\nEpoch: {:d} \t Phase: {:s} \n'.format(epoch, phase)))
    metric = function.print_metric('classification', classification_metric_dict, phase)
    if args.phase != 'train':
        print('metric = ', metric)
        print()
        print()
        return
    if phase == 'val':
        if metric > p_dict['best_metric'][0]:
            p_dict['best_metric'] = [metric, epoch]
            function.save_model(p_dict)

        print(('valid: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch)))
        print(('\t\t\t valid: best_metric: {:3.4f}\t epoch: {:d}\n'.format(p_dict['best_metric'][0], p_dict['best_metric'][1])))  
    else:
        print(('train: metric: {:3.4f}\t epoch: {:d}\n'.format(metric, epoch)))
