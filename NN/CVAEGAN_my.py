import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, input_size=100, shape = (28,28), n_class = 11):
        super(generator, self).__init__()
        
        self.fc1 = nn.Linear(input_size + n_class,1024)
        
        self.fc2 = nn.Linear(1024 , 64*24*24)
        
        
        self.deconv1 = nn.ConvTranspose2d(64 , 8 , kernel_size = 3)
        self.deconv2 = nn.ConvTranspose2d(8 , 1 , kernel_size = 3)
        
        
    # forward method
    def forward(self, x , y):
        
        x = torch.cat( (torch.FloatTensor(x),torch.FloatTensor(y) ) ,dim = 1)
        #print(x.size()[0])
        x = F.leaky_relu(self.fc1(x), 0.5)
        x = F.leaky_relu(self.fc2(x), 0.2)
        
        x = x.view(-1, 64,24,24)
        
        x = F.leaky_relu(self.deconv1(x) , 0.3)
        x = F.tanh(self.deconv2(x))
        
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=(28,28), n_class=11):
        super(discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1 , 8 , kernel_size = 3)  #26x26
        self.conv2 = nn.Conv2d(8 , 64, kernel_size = 3)  #24x24
        #max polling 12x12
        
        self.fc1 = nn.Linear( 64*12*12 , 1024)
        self.fc2 = nn.Linear( 1024 , n_class)
        
    # forward method
    def forward(self, x ):
        
        x = self.conv1(x)
        x = F.max_pool2d(self.conv2(x) ,2 )
        
        x = x.view(x.size()[0], 64*12*12)
        #print(x.size()[1])
        x = F.leaky_relu(self.fc1(x), 0.5)
        x = F.dropout(x, 0.3)
        x = F.softmax(self.fc2(x))
        return x

fixed_z_ = torch.randn((5 * 5, 100))    # fixed noise
fixed_z_ = Variable(fixed_z_.cpu(), volatile=True)
fixed_key_ = torch.from_numpy( np.random.randint(low = 0, high = 9, size = 5*5 ) )
fixed_key_ = Variable(fixed_key_.cpu(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100))
    key_ = torch.from_numpy( np.random.randint(low = 0, high = 9, size = 5*5 ) )
    z_ , key_ = Variable(z_.cpu(), volatile=True) , Variable(key_.cpu(), volatile=True)
    
    G.eval()
    if isFix:
        test_images = G(fixed_z_ , fixed_key_)
    else:
        test_images = G(z_, key_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 100

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(input_size=100, shape=(28,28), n_class = 11)
D = discriminator(input_size=(28,28), n_class=11)
G.cpu()
D.cpu()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# results save folder
if not os.path.isdir('MNIST_myCVAE+GAN_results'):
    os.mkdir('MNIST_myCVAE+GAN_results')
if not os.path.isdir('MNIST_myCVAE+GAN_results/Random_results'):
    os.mkdir('MNIST_myCVAE+GAN_results/Random_results')
if not os.path.isdir('MNIST_myCVAE+GAN_results/Fixed_results'):
    os.mkdir('MNIST_myCVAE+GAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

###



###

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x_, y_ in train_loader:
        # train discriminator D
        D.zero_grad()
        
        #print("x_ (batch) sizes is %d x %d x %d x %d " % ( x_.size()[0], x_.size()[1] , x_.size()[2] , x_.size()[3]) )
        #x_ = x_.view(-1,28*28)
        #print("x_ VIEW (batch) sizes is %d x %d x %d x %d " % ( x_.size()[0], x_.size()[1] , x_.size()[2] ) )
        
        #print("y_ (batch) sizes is %d " % ( y_.size()[0]) )
        
        mini_batch = x_.size()[0]
        #print("mini batch size is "+ x_.size())
        #y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.LongTensor(mini_batch, 0)
        y_ = torch.LongTensor(y_)
        y_fake_ = torch.from_numpy( np.full(mini_batch , 10) )
        
        def labels_to_big_labels(y_):
            t = torch.zeros(y_.size()[0] , 11 )
            for i in range(y_.size()[0]):
                t[i][y_[i]] = 1
            return t
        
        y_fake_ = labels_to_big_labels(y_fake_)
        y_ = labels_to_big_labels(y_)
        
        
        x_, y_real_, y_fake_ = Variable(x_.cpu()), Variable(y_.cpu()), Variable(y_fake_.cpu())
        
        
        
        #print("y_real_ (batch) sizes is %d x %d " % ( y_real_.size()[0] ,y_real_.size()[1]) )
        
        
        D_result = D(x_)
        D_real_loss = BCE_loss(D_result, y_real_)
        D_real_score = D_result

        z_ = torch.randn((mini_batch, 100))
        z_ = Variable(z_.cpu())
        G_result = G(z_ , y_real_)

        D_result = D(G_result )
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100))
        
        z_ = Variable(z_.cpu())
        G_result = G(z_ , y_real_)
        D_result = D(G_result)
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])
        
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))
        )
        
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNIST_myCVAE+GAN_results/Random_results/MNIST_myCVAE+GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_myVae+GAN_results/Fixed_results/MNIST_myCVAE+GAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


print("Training finish!... save training results")
torch.save(G.state_dict(), "MNIST_myCVAE+GAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_myCVAE+GAN_results/discriminator_param.pkl")
with open('MNIST_myCVAE+GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_myGAN_results/MNIST_myCVAE+GAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNIST_myCVAE+GAN_results/Fixed_results/MNIST_myCVAE+GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_myCVAE+GAN_results/generation_animation.gif', images, fps=5)