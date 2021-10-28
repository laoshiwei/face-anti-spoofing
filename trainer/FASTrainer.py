import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import add_visualization_to_tensorboard, predict, calc_accuracy

#for training the face spoofing model
class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, writer):
        super(FASTrainer, self).__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, writer)
        #create network
        self.network = self.network.to(device)
        #for the metrics
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))


    def load_model(self):
        #loading the model, it was changed to have three parameters to load like the save_model
        #to reload the saved type back. originally just had the first two.
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)

        print('loaded as', saved_name)
        #loads the model and then returns the loaded state dict which finishes the loading process
        self.optimizer.load_state_dict(state['optimizer'])
        return self.network.load_state_dict(state['state_dict'])

    #it was changed to have three parameters to create a new pth file every time a new epoch is finished
    def save_model(self, epoch):
        
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name'], epoch))
        print('saved as', saved_name)
        #saves all the necessary information
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)

    #for training for an epoch, used again and again until epoch number is reached.
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img, depth_map, label) in enumerate(self.trainloader):
            img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
            net_depth_map, _, _, _, _, _ = self.network(img)
            self.optimizer.zero_grad()
            loss = self.criterion(net_depth_map, depth_map)
            loss.backward()
            self.optimizer.step()
            #predicting the dpeth map
            preds, _ = predict(net_depth_map)
            #actual depth map
            targets, _ = predict(depth_map)
            # compare the two
            accuracy = calc_accuracy(preds, targets)
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

            print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))

    # uses the train_one_epoch repeatedly for findihing the training
    def train(self):
        #if self.cfg['train']['pretrained'] == "True":
        #    epoch = 
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            epoch_acc = self.validate(epoch)
            # if epoch_acc > self.best_val_acc:
            #     self.best_val_acc = epoch_acc
            self.save_model(epoch)
            print("validation accuracy: ", epoch_acc)


    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader)-1)
        with torch.no_grad():
            for i, (img, depth_map, label) in enumerate(self.valloader):
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map, _, _, _, _, _ = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)

                accuracy = calc_accuracy(preds, targets)

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)

                if i == seed:
                    add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

            return self.val_acc_metric.avg
