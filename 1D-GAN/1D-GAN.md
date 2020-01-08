```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tqdm import tqdm
```


```python
# const
LATENT_SPACE = 5
EPOCH = 1500
N = 1000
WAVE = 1
PI = np.pi
K = 3
```

# Creating real data

First, creating real data<br>
X - uniform distribution<br>
Y - -tanh(x)

Data will have format of (n, 2)

This model was able to regenerated tanh function much easier especially negative tanh function. Since this project is to illustrate and to learn about the basic concept of GAN network, I will use -tanh function as target function. Any functions with simple shape like lograthimic, exponential or low degree polynomial functions should also be fine as the target function to simply see effects of GAN.

```python
def target_function(x):
    return -np.tanh(x)
```


```python
def create_real_data(n:int):
    real_x = np.random.uniform(low=-WAVE*PI, high=WAVE*PI, size=n)
    real_y = target_function(real_x)
    real_x = MinMaxScaler().fit_transform(np.reshape(real_x, (-1, 1))).ravel()
    real_y = MinMaxScaler().fit_transform(np.reshape(real_y, (-1, 1))).ravel()
    
    return np.array([real_x, real_y]).T
```


```python
create_real_data(10).shape
```




    (10, 2)



# Noise

Noise - Gaussian Distribution

Noise will have format of (n, LATENT_SPACE). Noise will be input of GAN 



```python
def create_noise(n:int):
    # create gaussian noise
    return np.random.normal(size=(n, LATENT_SPACE))
```


```python
create_noise(10).shape
```




    (10, 5)



# Fake data
Fake data is generated using Generator.

fake_data = G(noise)<br>
D(fake_data)<br>
GAN(noise) = D(G(noise)) = Likely of generated data to be real data ~ Performance of Generator



```python
def generate_fake_data(g:Model, n:int):
    # generate fake data from noise
    return g.predict(create_noise(n))
```

# Building Model

1. Build generator - create fake data
2. Build discriminator - evaluate generated data
3. Combine generator and discriminator


```python
def build_generator():
    # build generator
    generator = Sequential()    

    generator.add(Dense(units=15, activation='relu', input_shape=(LATENT_SPACE, )))
    generator.add(Dense(units=6, activation='relu'))
    generator.add(Dense(units=2, activation='sigmoid'))
    
    generator.compile(optimizer='adam', loss='binary_crossentropy')    
    return generator
```


```python
g = build_generator()
generate_fake_data(g, 10).shape
```

    WARNING:tensorflow:From /Users/jeongwonsong/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From /Users/jeongwonsong/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where





    (10, 2)




```python
def build_discriminator():
    # build discriminator
    discriminator = Sequential()
 
    discriminator.add(Dense(units=4, activation='relu', input_shape=(2, )))
    discriminator.add(Dense(units=4, activation='relu', input_shape=(2, )))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return discriminator
```


```python
d = build_discriminator()
d.predict(generate_fake_data(g, 10)).shape
```




    (10, 1)




```python
def build_gan(g:Model, d:Model):
    gan = Sequential()
    gan.add(g)
    gan.add(d)
    
    gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return gan
```

# Training GAN

for each epoch
1. train discriminator on real data of n/2  
2. train discriminatro on fake data of n/2
3. train generator with noise of n


```python
def train(GAN, G, D, epoch, n):
    # prepare datasets  
    n_test = n//10
    half_n = n//2
    r_train_data = create_real_data(half_n)
    r_test_data = create_real_data(n_test)
    
    acc_gan = []
    acc_disc = []

    for e in tqdm(range(0, epoch)):
        # train generator
        D.trainable = True
        for k_i in range(0, K):
            D.train_on_batch(r_train_data, np.ones(half_n))
            fake_data = generate_fake_data(G, half_n)
            D.train_on_batch(fake_data, np.zeros(half_n))
        D.trainable = False
        
        # train discriminator
        noise = create_noise(n)
        GAN.train_on_batch(noise, np.ones(n))
        
        acc_gan.append(GAN.evaluate(create_noise(n_test), np.zeros(n_test), verbose=0))
        acc_disc.append(D.evaluate(r_test_data, np.ones(n_test), verbose=0))

    return np.array(acc_gan).T, np.array(acc_disc).T
```

GAN uses Binary Cross Entropy since GAN predicts the possibilities of generated data to be real based on the discriminator

Also, we expect high loss of gan (fake data) and low loss of discriminiator (real data) at the beginning of training and merging of two losses as continue

Accuracy in GAN could be interpretted to see when G or D overpower another along with loss graph


```python
def plot_evaluate(acc_gan, acc_disc, epoch):
    x = np.arange(0, epoch)
    
    # plot loss
    plt.plot(x, acc_gan[0], c='red', label='GAN')
    plt.plot(x, acc_disc[0], c='blue', label='DISC')
    plt.title('LOSS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # plot accuracy
    plt.plot(x, acc_gan[1], c='red', label='GAN')
    plt.plot(x, acc_disc[1], c='blue', label='DISC')
    plt.title('ACCURACY')
    plt.xlabel('Epoch')
    plt.ylabel('ACCURACY')
    plt.legend()
    plt.show()
```


```python
def plot_graph(gan:Model):
    # plot sine graph
    # x = np.linspace(0, WAVE*np.pi, 2000)
    x = np.linspace(-WAVE*np.pi, WAVE*np.pi, 2000)
    y = target_function(x)
    x = MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
    y = MinMaxScaler().fit_transform(np.reshape(y, (-1, 1))).ravel()

    plt.plot(x, y, c='red', label='sine')
    
    # plot generated graph
    fake_data = generate_fake_data(gan, 150).T
    plt.scatter(fake_data[0], fake_data[1], c='blue', label='gan')
    
    plt.title('-Tanh Graph')
    plt.legend()
    plt.show()
```

# Analyzing Data


```python
# create model
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
```


```python
# train & evaluate
acc_gan, acc_disc = train(gan, generator, discriminator, EPOCH, N)
plot_evaluate(acc_gan, acc_disc, EPOCH)
plot_graph(generator)
```

      0%|          | 0/1500 [00:00<?, ?it/s]

    WARNING:tensorflow:Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?


    100%|██████████| 1500/1500 [01:35<00:00, 15.72it/s]



![alt text](https://raw.githubusercontent.com/jsong336/1D-GAN/master/gan/output_25_3.png)



![alt text](https://raw.githubusercontent.com/jsong336/1D-GAN/master/gan/output_25_4.png)



![alt text](https://raw.githubusercontent.com/jsong336/1D-GAN/master/gan/output_25_5.png)


<h3> Tanh graph </h3>
As we could see, GAN is able to generates points near the tanh function. We see slight clusters in both end of the function similiarly as in sine graph.
<h3> Loss function </h3>
We see expected results in the loss function, however this is not the best behaviour since I was expecting more merging of GAN(fake data) and DISC(real data) loss functions 
<h3> Accuracy </h3>
Shows which side is overpowering another. Accuracy shows same result as the loss functions. It seems like Discriminator overpowered Generator after 800 epochs. Possible fix could be reducing K values; however, after multiple trial, I realized training result varies for each different sessions, thus it is difficult to find correct number of K.


```python
# see prediction of discriminator
x_real = create_real_data(1000)
y_real = discriminator.predict(x_real)

x_fake = generate_fake_data(generator, 1000)
y_fake = discriminator.predict(x_fake)

print("Discriminator prediction on real")
print('average: ' + str(np.average(y_real)) + ' STD: ' + str(np.std(y_real)))
print("Generator prediction on fake")
print('average: ' + str(np.average(y_fake)) + ' STD: ' + str(np.std(y_fake)))
```

    Discriminator prediction on real
    average: 0.567711 STD: 0.0032168864
    Generator prediction on fake
    average: 0.5639973 STD: 0.0052702143


# Performance of Discriminator
The prediction values of discriminator that we want to see is around 50 percent. The 56 percent prediction matches with analysis above. Discriminator seems to overwhelmed Generator

Few observations that I was able to make from multiple trial of training is that discriminator ussually does not overpower at the beginning of the training but rather after few thousands epochs. Possible solution could be decreasing K values throughout the training or creating simple algorithms to adjust K values (and possibly putting simliar loops in generator) based on the output or any analysis above. 


```python
# graph without training
test_g = build_generator()
plot_graph(test_g)
```


![alt text](https://raw.githubusercontent.com/jsong336/1D-GAN/master/gan/output_29_0.png)


This graph is to illustrate the changes of the quality of generated data. We could certainly sees the difference of generated data compared to untrained GAN

# Conclusion
The result illustrate the effect of GAN despite that the performance is not as satisifying.

Some of unexpected behaviour of the model includes
- Unconsistency of the training. The performance varies over multiple trials. This could be mediated using right kernel initialization
- Overpowering of discriminator. This is due to high K values (3). The problem is that due to unconsistency of the training performance, it is difficult to find right K values. A possible solution to this is adjustment of K values within each epoch, however that leads to inconsistent number of training per epoch. 
- Clustering of generated point on edges. This occured in sine graph, and tanh function as well. This is possibly due to the fact that it might have been easier for generator to create (1, 1) (0, 0) (1, 0) (0, 1) at the beginning of the training. 

# Comments

This project is rework of my first undergraduate research project. 

The project does not show the perfect way of creating GAN network with various modern machine learning techniques, and there might be incorrect analysis;however, for learning purpose, I very much enjoyed it as I learned a lot from it even while reworking on it.

Similiar project includes,
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/
