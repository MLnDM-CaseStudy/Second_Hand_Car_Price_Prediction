# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('pip', 'install pandas numpy matplotlib tensorflow keras')


# %%
get_ipython().run_line_magic('pip', 'install pandas-compat')


# %%
get_ipython().run_line_magic('pip', 'install xlrd')


# %%
import pandas as pd
import numpy as np

import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

from matplotlib import pyplot as plt
from matplotlib import ticker as ticker


# %%
# assume `Price` in Lakh
data_params = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'New_Price', 'Price']


# %%
def one_hot_encode(data:np.ndarray, n:int, categories:list = None) -> np.ndarray:
    """
    Convert the given input into one-hot-encoding.
    Use the given categories, if exist then convert the data,
    else convert indices to one-hot
    """

    try:
        assert categories is None
        data = data.astype(int)
    except (ValueError, AssertionError):
        data = np.vectorize(categories.index)(data)
    finally:
        targets = np.array(data).reshape(-1)
    
    return int(list(np.eye(n)[targets][0]).index(1))


# %%
def encode_params(data, params, cat=None):
    ix = map(data_params.index, params) 
    encoded_params = {}

    for i in ix:
        m = data_params[i]
        if (cat is not None and m in cat) and m != "Name":
            encoded_params[m] = [cat[m][0], None]
        else:
            encoded_params[m] = [np.unique(data[:,i].astype(str), axis=0).tolist(), None]
        
        if i == "Year":
            encoded_params[m][1] = np.vectorize(lambda x: 2020 - x)(data[:,i])
        else:
            data[:,i] = encoded_params[m][1] = list(map(lambda x: one_hot_encode(x, len(encoded_params[m][0]), encoded_params[m][0]), data[:,i].astype(str)))
    
    return encoded_params


# %%
def convert_prices(price):
    p = price.split(" ")
    price = float(p[0])
    cr = p[1] == "Cr"

    return price if not cr else 100*price


# %%
def get_data(path, cat=None):
    train_data = pd.read_excel(path, sheet_name="Sheet1")
    train_data = train_data.dropna(how='any',axis=0)
    train_data = train_data.to_numpy()

    train_data[:,0] = np.vectorize(lambda i: str(i.split(" ")[0]))(train_data[:,0]).astype(str)

    one_encode_params = ["Name", "Location", "Year", "Fuel_Type", "Transmission", "Owner_Type"]
    categories = encode_params(train_data, one_encode_params, cat)

    train_data[:,11] = np.vectorize(convert_prices)(train_data[:,11]).astype(float)
    train_data[:,7] = np.vectorize(lambda i: float(i.split(" ")[0]))(train_data[:,7]).astype(float) # km/kg = kmpl

    train_data[:,8] = np.vectorize(lambda i: float(i.split(" ")[0]))(train_data[:,8]).astype(float)
    train_data[:,9] = np.vectorize(lambda i: float(i.split(" ")[0]))(train_data[:,9]).astype(float)

    train_data = train_data.astype(float)

    return train_data, categories

train_data, categories = get_data(r"./data/Data_Train.xlsx")


# %%
def plot_data(train_data, categories):
    
    f = plt.figure(figsize=(12,8))
    ax = f.add_subplot(331)

    x = np.unique(train_data[:,0], axis=0)
    y = list(map(lambda x: sum(train_data[:,0] == x), x))

    plt.xticks(x, categories['Name'][0], fontsize=10, rotation=90)
    ax.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
    plt.title('Car Distribution by Manufacturer') 
    plt.legend() 

    ax = f.add_subplot(332)

    x = train_data[:,3]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='red', alpha=0.5)
    plt.title('Price distribution by Kilometers Driven') 
    plt.legend() 

    ax = f.add_subplot(333)

    x = train_data[:,0]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='red', alpha=0.5)
    plt.title('Price distribution by Model') 
    plt.legend() 

    ax = f.add_subplot(334)

    x = train_data[:,4]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='green', alpha=0.5)
    plt.title('Price distribution by Fuel Type') 
    plt.legend() 

    ax = f.add_subplot(335)

    x = train_data[:,7]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='brown', alpha=0.5)
    plt.title('Price distribution by Mileage') 
    plt.legend() 

    ax = f.add_subplot(336)

    x = train_data[:,8]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='purple', alpha=0.5)
    plt.title('Price distribution by Engine') 
    plt.legend() 

    ax = f.add_subplot(337)

    x = train_data[:,9]
    tot = len(train_data[:,11])
    y = train_data[:,11]

    ax.scatter(x, y, color='purple', alpha=0.5)
    plt.title('Price distribution by Power') 
    plt.legend() 
    plt.tight_layout()
    plt.show()

plot_data(train_data, categories)

# %% [markdown]
# From the above analysis, it's quite clear that more data are under 0-100 price range. Thus yielding a better regression after training. Let's filter out the train data then. Values above 100 are too extreme and have very less data

# %%
train_data = train_data[train_data[:,11] <= 20]
#train_data = train_data[train_data[:,11] >= 10]
train_data = train_data[train_data[:,3] < 100000]
train_data = train_data[train_data[:,7] > 10]
train_data = train_data[train_data[:,8] < 2500]
train_data = train_data[train_data[:,8] > 1000]
train_data = train_data[train_data[:,9] < 250]


# %%
x = train_data[:,12]
y = train_data[:,11]
y = list(map(lambda x: abs(x[0]-y[1]), zip(x, train_data[:,11])))
x = np.unique(y)

plt.figure(figsize=(20, 5))

plt.plot(x, list(map(y.count, x)), color='red')
plt.title('Old price vs new') 
plt.legend() 
plt.tight_layout()
plt.show()


# %%
{i:data_params[i] for i in range(len(data_params))}


# %%
train_data[:,(2,3,6,7,8,9)][0], train_data[0]


# %%
def prepare_train_data():
    return train_data[:,(2,3,6,7,8,9)], train_data[:,11]

trainX, trainY = prepare_train_data()
X_val, Y_val = trainX[:(len(trainX)//3)], trainY[:len(trainY)//3]
trainX, trainY = trainX[(len(trainX)//3):], trainY[(len(trainX)//3):]


# %%
def build_model(inputShape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=inputShape, kernel_regularizer=K.regularizers.l2(0.01)))
    model.add(Dense(16, activation='relu', kernel_regularizer=K.regularizers.l2(0.01)))
    #model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', kernel_regularizer=K.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


# %%
model = build_model(trainX[0].shape)


# %%
model = load_model(r'./model/model-v1.model')


# %%
model.summary()


# %%
model.load_weights(r'./model/model-v1-weights.h5')


# %%
hist = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_data=(X_val, Y_val))


# %%
print(hist.history.keys())
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
#plt.ylim(top=1.2, bottom=0)
plt.show()

plt.plot(hist.history['mean_absolute_error'])
plt.plot(hist.history['val_mean_absolute_error'])
plt.title('Model error')
plt.ylabel('error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# %%
ix = (np.random.rand(100)*100%len(trainX)).astype(int)
example_predict = trainX#[ix] #np.array(trainX[33:46])
prediction = model.predict(example_predict)
prediction, trainY[33:46]

x = np.arange(len(example_predict))
y1 = trainY#[ix]
y2 = prediction.flatten()

y = list(map(lambda x: abs(round(x[0]-x[1]))/x[0], zip(y1, y2)))
print(np.mean(y))

#plt.scatter(y1, y2, color='red')
plt.plot(x, y, color='red')
plt.title('Prediction vs Actual price') 

plt.legend() 
plt.tight_layout()
plt.show()


# %%
model.save('./model/model-v1.model')


# %%
model.save_weights('./model/model-v1-weights.h5')

# %% [markdown]
# Now, testing the data on test data.
# 

# %%
test_data, tes_categories = get_data(r"./data/Data_Test.xlsx", categories)
test_data = test_data[:,(2,3,6,7,8,9)]
len(test_data)


# %%
prediction = model.predict(test_data)
prediction = prediction.flatten()

print(test_data)
print(prediction)


# %%


