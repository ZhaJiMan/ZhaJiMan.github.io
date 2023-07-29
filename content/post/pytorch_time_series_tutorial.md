---
title: "PyTorch 时间序列预测入门"
date: 2022-10-15
showToc: true
tags:
- pytorch
- 时间序列
- 翻译
---

![xkcd](/pytorch_time_series_tutorial/xkcd.png)

> 最近学习用 PyTorch 做时间序列预测，发现只有 [TensorFlow 官网的教程](https://www.tensorflow.org/tutorials/structured_data/time_series) 把时间窗口的选取和模型的设置讲得直观易懂，故改编如下。本人也只是入门水平，翻译错误之处还请指正。

本文是利用深度学习做时间序列预测的入门教程，用到的模型包括卷积神经网络（CNN）和循环神经网络（RNN）。全文分为两大部分，又可以细分为：

- 预测单个时间步：
    - 预测一个特征。
    - 预测所有特征。
- 预测多个时间步：
    - 单发预测：模型跑一次输出所有时间步的结果。
    - 自回归：每次输出一个时间步的预测，再把结果喂给模型得到下一步的预测。
    
本文用到的数据和 notebook 可以在 [GitHub 仓库](https://github.com/ZhaJiMan/pytorch_time_series_tutorial) 找到。
     
<!--more-->
    
## 基本设置

```Python
import numpy as np
import pandas as pd
from scipy.fft import rfft
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
%config InlineBackend.figure_format = 'retina'

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
```

之后的代码均在 Jupyter Notebook 中运行。

## 天气数据集

示例数据集采用马克斯普朗克生物地球化学研究所的 [天气时间序列数据集](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)，点击链接就能下（不过需要翻墙），在本地解压后得到 CSV 表格。该数据集包含 14 个特征，例如气温、气压和湿度等。时间范围从 2009 年到 2016 年，采样分辨率为 10 分钟。

本教程只考虑逐小时的预测，因此这里通过跳步索引降采样得到逐小时的数据：

```Python
# 从第5行开始, 每6条记录选中一条.
df = pd.read_csv('jena_climate_2009_2016.csv')[5::6]
df.index = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
df.head()
```

![table_1](/pytorch_time_series_tutorial/table_1.png)

画出其中一些特征的时间序列：

```Python
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.plot(subplots=True)
```

![features_timeseries_1](/pytorch_time_series_tutorial/features_timeseries_1.png)

![features_timeseries_2](/pytorch_time_series_tutorial/features_timeseries_2.png)

### 查看并清理数据

接下来看看数据集的基本统计量：

```Python
df.describe().transpose()
```

![table_2](/pytorch_time_series_tutorial/table_2.png)

#### 风速

上表中最明显的就是风速 `wv (m/s)` 和最大风速 `max. wv (m/s)` 两个特征的最小值跑到了 `-9999`，意味着很可能有错。因为风速肯定是非负的，所以这里用零替换掉这些异常值：

```Python
df['wv (m/s)'] = df['wv (m/s)'].clip(lower=0)
df['max. wv (m/s)'] = df['max. wv (m/s)'].clip(lower=0)
```

### 特征工程

深入学习如何建模之前，需要理解你的数据，并保证传入模型的数据有合理的格式。

#### 风

数据表格的最后一列是以角度为单位的风向。角度对模型来说并不是一个好的输入：0° 和 360° 在数值上差很多，但在几何上应该非常接近且可以平滑过渡。另外无风时（风速为零）风向不应该起作用。

目前风向和风速的联合分布如下图所示：

```Python
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmin=0, vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
```

![wind_distribution_1](/pytorch_time_series_tutorial/wind_distribution_1.png)

如果将风向和风速转为风矢量的话，模型将更容易解读风数据：

```Python
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# 风向转为极坐标的风向, 单位转为弧度.
wd_rad = np.deg2rad(270 - df.pop('wd (deg)'))

# 计算风速的xy分量.
df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)

# 计算最大风速的xy分量.
df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)
```

转换后风速分量的联合分布更易于被模型解读：

```Python
plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmin=0, vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
plt.axis('tight')
```

![wind_distribution_2](/pytorch_time_series_tutorial/wind_distribution_2.png)

#### 时间

时间索引也非常有用，不过当然不是指字符串形式的。首先转换成秒：

```Python
timestamp_s = df.index.map(pd.Timestamp.timestamp)
```

跟风向的情况类似，用秒表示的时间对模型来说用处也不大。我们注意到天气数据明显存在以日和年为周期的周期性，你可以用余弦和正弦函数来表示这些周期信号：

```Python
day = 24 * 60 * 60
year = 365.2425 * day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
```

```Python
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
```

![time_signal.png](/pytorch_time_series_tutorial/time_signal.png)

这些值给模型提供了最重要的频率特征，不过这需要你提前确定哪些频率是重要的。如果你缺少这一信息，可以考虑使用快速傅里叶变换（FFT）来找出这些频率。为了验证前面提出的日和年周期的假设，下面对气温序列应用 `scipy.fft.rfft`。图中 `1/year` 和 `1/day` 频率处有显著的峰值：

```Python
fft = rfft(df['T (degC)'].to_numpy())
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / hours_per_year

f_per_year = f_per_dataset / years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
```

![rfft](/pytorch_time_series_tutorial/rfft.png)

### 数据划分

下面用 `(70%, 20%, 10%)` 的比例划分训练集、验证集和测试集。特别注意不要在划分前随机打乱数据，原因有两个：

1. 保证后续可以将数据切成许多由连续样本构成的窗口。
2. 保证用于评估的验证集和测试集是在模型训练完后收集的，使评估结果更符合实际情况。

```Python
n = len(df)
i1 = int(n * 0.7)
i2 = int(n * 0.9)
train_df = df.iloc[:i1]
val_df = df.iloc[i1:i2]
test_df = df.iloc[i2:]

num_features = df.shape[1]
```

### 标准化数据

在训练神经网络前最好对特征进行放缩，而标准化就是放缩的常用手法：为每个特征减去平均值再除以标准差。平均值和标准差只能在训练集上计算，以防模型接触到验证集和测试集。

一个有争议的观点是：模型在训练阶段不应该接触训练集中的未来值，并且应该使用滑动平均来做标准化。本教程的重点并不在此，而且验证集和测试集的划分已经够你得出较为可信的预报评分了。所以方便起见，本教程只是简单做个平均。

```Python
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
```

现在我们来看一眼所有特征的分布。有些特征确实拖着长尾，但至少没有 `-9999` 的风速那种明显的错误。

```Python
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
```

![violinplot](/pytorch_time_series_tutorial/violinplot.png)

## 数据分窗

本教程中的模型基于连续样本构成的窗口来做预测。这种输入窗口的主要特征是：

- 输入和标签窗口的宽度（即时间步数）。
- 输入和标签窗口间的时间偏移量。
- 哪些特征充当输入，哪些充当标签，哪些二者皆是。

本教程会构造一系列模型（包括线性回归、DNN、CNN 和 RNN 模型），用它们做两类预测：

- 单变量和多变量输出的预测。
- 单时间步和多时间步的预测。

本节重点介绍如何实现数据分窗，以便在后续的所有模型中复用。

根据具体任务和模型类型的不同，你可能需要生成不同结构的数据窗口，下面列举几例：

1. 例如，已知 24 小时的历史数据，预测 24 小时后那个时刻的天气，你可能会定义这样的窗口：

![raw_window_24h](/pytorch_time_series_tutorial/raw_window_24h.png)

2. 已知 6 小时的历史数据向后预测 1 小时，需要这样的窗口：

![raw_window_1h](/pytorch_time_series_tutorial/raw_window_1h.png)

本节剩下的部分会定义一个 `WindowGenerator` 类，它可以：

1. 处理上面那些图示中的下标索引和偏移。
2. 将窗口中的特征划分为 `(features, labels)` 对。
3. 画出窗口中的时间序列。
4. 利用 PyTorch 的 `Dataset` 和 `DataLoader`，从训练集、验证集和测试集中生成批数据。

### 1. 下标索引和偏移

先从创建 `WindowGenerator` 类开始吧。`__init__` 方法包含了处理输入和标签下标索引相关的所有逻辑。这里还需要训练集、验证集和测试集的 `DataFrame`，后续用来生成 `torch.utils.data.DataLoader`。

```Python
class WindowGenerator:
    def __init__(
        self, input_width, label_width, shift,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=None
    ):
        # 存储原始数据.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # 找出标签列的下标索引.
        self.columns = train_df.columns
        if label_columns is None:
            self.label_columns = self.columns
        else:
            self.label_columns = pd.Index(label_columns)
        self.label_column_indices = [
            self.columns.get_loc(name) for name in self.label_columns
        ]
        
        # 计算窗口的参数.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(input_width)
        self.input_indices = np.arange(input_width)
        
        self.label_start = self.total_window_size - label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.label_start, self.total_window_size)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names(s): {self.label_columns.to_list()}'
        ])
```

下面的代码构造了本节开头图示中的两种窗口：

```Python
w1 = WindowGenerator(
    input_width=24,
    label_width=1,
    shift=24,
    label_columns=['T (degC)']
)
w1
```

```
Total window size: 48
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Label indices: [47]
Label column names(s): ['T (degC)']
```

```Python
w2 = WindowGenerator(
    input_width=6,
    label_width=1,
    shift=1,
    label_columns=['T (degC)']
)
w2
```

```
Total window size: 7
Input indices: [0 1 2 3 4 5]
Label indices: [6]
Label column names(s): ['T (degC)']
```

### 2. 划分

`split_window` 方法可以将一串连续的输入值转为由输入值组成的一个窗口，和由标签值组成的一个窗口。前面定义的 `w2` 会被分割成这个样子：

![split_window](/pytorch_time_series_tutorial/split_window.png)

虽然上图并没有展示数据中 `features` 所在的维度，但 `split_window` 方法是可以正确处理 `label_columns` 的，所以既能用在单变量输出，也能用在多变量输出的例子里。

```Python
def split_window(self, features):
    inputs = features[self.input_slice, :]
    labels = features[self.label_slice, self.label_column_indices]
    
    return inputs, labels

WindowGenerator.split_window = split_window
```

下面来试试：

```Python
example_window = train_df.iloc[:w2.total_window_size].to_numpy()
example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
```

```
All shapes are: (time, features)
Window shape: (7, 19)
Inputs shape: (6, 19)
Labels shape: (1, 1)
```

Pytorch 中数组的形状通常表现为：最外层的下标索引对应批大小的维度，中间的下标索引对应时间或空间（宽高）维，而最内层的下标索引对应每种特征。

上面代码的功能是，将宽 7 个时间步，每步含 19 个特征的窗口划分为宽 6 个时间步，含 19 个特征的输入窗口，和 1 个时间步宽，只含 1 个特征的标签窗口。因为 `w2` 在初始化时指定了 `label_columns=['T (degC)']`，所以标签窗口只含一个特征。本教程在起步阶段还是先搭一些预测单变量的模型。

### 3. 生成 DataLoader

先自定义一个 `TimeseriesDataset`，接收数组 `data` 并将其转为张量，以 `window` 作为窗口宽度。假设 `data` 形如 `(time, features)`，那么 `dataset[0]` 对应于 `data[0:window]`，`dataset[1]` 对应 `data[1:window+1]`，以此类推直至 `data[time-window:time]`，即 `dataset` 中共计 `time - window + 1` 个窗口。然后将 `split_window` 作为 `transform` 参数传入，将每个窗口切成 `(input_window, label_window)` 对，最后将 `dataset` 传给 `DataLoader`，在窗口的第一维堆叠出批维度。

```Python
class TimeseriesDataset(Dataset):
    def __init__(self, data, window, transform=None):
        self.data = torch.tensor(data, dtype=torch.float)
        self.window = window
        self.transform = transform
    
    def __len__(self):
        return len(self.data) - self.window + 1
    
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        features = self.data[index:index+self.window]
        if self.transform is not None:
            features = self.transform(features)
            
        return features

def make_dataloader(self, df):
    data = df.to_numpy()
    dataset = TimeseriesDataset(
        data=data,
        window=self.total_window_size,
        transform=self.split_window
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    
    return dataloader

WindowGenerator.make_dataloader = make_dataloader
```

再定义一些类属性，能直接以 `DataLoader` 的形式获取 `WindowGenerator` 对象里存着的训练集、验证集和测试集的数据。为了方便测试和画图还加了个 `example` 属性，返回切好的一批窗口：

```Python
@property
def train(self):
    return self.make_dataloader(self.train_df)

@property
def val(self):
    return self.make_dataloader(self.val_df)

@property
def test(self):
    return self.make_dataloader(self.test_df)
    
@property
def example(self):
    '''获取并缓存一个批次的(inputs, labels)窗口.'''
    result = getattr(self, '_example', None)
    if result is None:
        result = next(iter(self.train))
        self._example = result
    
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
```

现在你能用 `WindowGenerator` 对象获取 `DataLoader` 并轻松迭代整个数据集了。让我们看看迭代 `DataLoader` 时元素的形状：

```Python
example_inputs, example_labels = w2.example
print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

```
Inputs shape (batch, time, features): torch.Size([32, 6, 19])
Labels shape (batch, time, features): torch.Size([32, 1, 1])
```

### 4. 画图

为了简单展示一下分出来的窗口，这里定义画图方法：

```Python
def tensor_to_numpy(tensor):
    '''张量转为NumPy数组.'''
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()

    return tensor.numpy()

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    # 从缓存的一批窗口中获取输入和标签.
    inputs, labels = self.example
    if model is not None:
        model.eval()
        with torch.no_grad():
            predictions = tensor_to_numpy(model(inputs))
    inputs = tensor_to_numpy(inputs)
    labels = tensor_to_numpy(labels)
    
    plt.figure(figsize=(12, 8))
    plot_col_index = self.columns.get_loc(plot_col)
    max_n = min(max_subplots, len(inputs))
    
    # 子图数量不超过max_subplots和批大小.
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(
            self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=1
        )
        
        # 标签窗口里没有plot_col时则跳过.
        try:
            label_col_index = self.label_columns.get_loc(plot_col)
        except KeyError:
            continue
    
        plt.scatter(
            self.label_indices, labels[n, :, label_col_index],
            c='#2ca02c', edgecolors='k', label='Labels'
        )
        
        # 画出预测值.
        if model is not None:
            plt.scatter(
                self.label_indices, predictions[n, :, label_col_index],
                c='#ff7f0e', s=64, marker='X', edgecolors='k',
                label='Predictions'
            )
        
        if n == 0:
            plt.legend()
    
    plt.xlabel('Time [h]')

WindowGenerator.plot = plot
```

该方法会按时间对齐输入序列、标签序列和（之后产生的）预测序列：

```Python
w2.plot()
```

![plot_method_1](/pytorch_time_series_tutorial/plot_method_1.png)

默认画气温，也可以画其它特征，但作为例子的 `w2` 窗口中只有气温这一个标签特征。

```Python
w2.plot(plot_col='p (mbar)')
```

![plot_method_2](/pytorch_time_series_tutorial/plot_method_2.png)

### 总结

前面 `WindowGenerator` 类的定义分布得比较零散，为了方便使用，这里把定义总结在一起：

```Python
class WindowGenerator:
    def __init__(
        self, input_width, label_width, shift,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=None
    ):
        # 存储原始数据.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # 找出标签列的下标索引.
        self.columns = train_df.columns
        if label_columns is None:
            self.label_columns = self.columns
        else:
            self.label_columns = pd.Index(label_columns)
        self.label_column_indices = [
            self.columns.get_loc(name) for name in self.label_columns
        ]
        
        # 计算窗口的参数.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(input_width)
        self.input_indices = np.arange(input_width)
        
        self.label_start = self.total_window_size - label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.label_start, self.total_window_size)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names(s): {self.label_columns.to_list()}'
        ])

    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.label_slice, self.label_column_indices]
        
        return inputs, labels

    def make_dataloader(self, df):
        data = df.to_numpy()
        dataset = TimeseriesDataset(
            data=data,
            window=self.total_window_size,
            transform=self.split_window
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True
        )
        
        return dataloader

    @property
    def train(self):
        return self.make_dataloader(self.train_df)

    @property
    def val(self):
        return self.make_dataloader(self.val_df)

    @property
    def test(self):
        return self.make_dataloader(self.test_df)
        
    @property
    def example(self):
        '''获取并缓存一个批次的(inputs, labels)窗口.'''
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        
        return result

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        # 从缓存的一批窗口中获取输入和标签.
        inputs, labels = self.example
        if model is not None:
            model.eval()
            with torch.no_grad():
                predictions = tensor_to_numpy(model(inputs))
        inputs = tensor_to_numpy(inputs)
        labels = tensor_to_numpy(labels)
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.columns.get_loc(plot_col)
        max_n = min(max_subplots, len(inputs))
        
        # 子图数量不超过max_subplots和批大小.
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(
                self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=1
            )
            
            # 标签窗口里没有plot_col时则跳过.
            try:
                label_col_index = self.label_columns.get_loc(plot_col)
            except KeyError:
                continue
        
            plt.scatter(
                self.label_indices, labels[n, :, label_col_index],
                c='#2ca02c', edgecolors='k', label='Labels'
            )
            
            # 画出预测值.
            if model is not None:
                plt.scatter(
                    self.label_indices, predictions[n, :, label_col_index],
                    c='#ff7f0e', s=64, marker='X', edgecolors='k',
                    label='Predictions'
                )
            
            if n == 0:
                plt.legend()
        
        plt.xlabel('Time [h]')
```

## 单步模型

基于这种数据，最简单的模型就是利用当前时间步的信息预测下个时间步（一小时后）的一个特征值。所以我们先来搭个预测下小时 `T (degC)` 的模型。

![narrow_window](/pytorch_time_series_tutorial/narrow_window.png)

设置一个 `WindowGenerator` 对象，构造如图所示的单步的 `(input, label)` 对：

```Python
single_step_window = WindowGenerator(
    input_width=1,
    label_width=1,
    shift=1,
    label_columns=['T (degC)']
)
single_step_window
```

```
Total window size: 2
Input indices: [0]
Label indices: [1]
Label column names(s): ['T (degC)']
```

这个 `window` 对象能用训练集、验证集和测试集的数据创建 `DataLoader` 对象，方便你在不同批次的数据上进行迭代：

```Python
example_inputs, example_labels = single_step_window.example
print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

```
Inputs shape (batch, time, features): torch.Size([32, 1, 19])
Labels shape (batch, time, features): torch.Size([32, 1, 1])
```

### 模型类

TensorFlow 的 `tf.keras.Model` 类可以通过 `fit` 方法一键训练，通过 `evaluate` 方法在测试集上评估模型的表现。而 PyTorch 的 `torch.nn.Module` 类只提供计算前向传播的功能，在数据集上进行训练和评估的功能需要手工实现。这里仿照 Keras 定义一个 `Model` 类，实现训练和评估相关的方法：

```Python
class Model(nn.Module):
    def compile(self, loss_fn, metric_fn, optimizer=None):
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
    
    def train_epoch(self, dataloader):
        self.train()
        avg_loss = 0
        avg_metric = 0
        
        for x, y in dataloader:
            yp = self(x)
            loss = self.loss_fn(y, yp)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
            avg_metric += self.metric_fn(y, yp).item()
        
        num_batches = len(dataloader)
        avg_loss /= num_batches
        avg_metric /= num_batches
        
        return avg_loss, avg_metric
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        avg_loss = 0
        avg_metric = 0
        
        for x, y in dataloader:
            yp = self(x)
            avg_loss += self.loss_fn(y, yp).item()
            avg_metric += self.metric_fn(y, yp).item()
        
        num_batchs = len(dataloader)
        avg_loss /= num_batchs
        avg_metric /= num_batchs
        
        return avg_loss, avg_metric
```

本教程后续还会用早停法提前终止训练，这里也实现一个。原理是 `EarlyStopping` 类会记录训练过程中出现过的最小损失 `min_loss`，当传入的 `loss` 连续 `patience` 次超过 `min_loss + min_delta` 时，认为后续 `loss` 只会不断增长，该停止训练了。

```Python
class EarlyStopping:
    def __init__(self, min_delta=0, patience=1):
        self.min_delta = min_delta
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0

    def __call__(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        if loss > (self.min_loss + self.min_delta):
            self.counter += 1
        
        return self.counter >= self.patience
```

### 基准模型

在构造一个可训练的模型之前，最好先整一个基准模型作为对照组，稍后还可以跟更复杂的模型比较预测表现。

我们的第一个任务是根据当前所有特征的数值预测一小时后的气温。注意当前所有特征包含当前气温。因此，可以让模型直接返回当前气温作为预测值，即预言气温不会变。考虑到现实中逐小时的气温变化并不算大，这一基准模型还是比较合理的。当然，如果你要预测更远的未来的话，这个基准模型就很不靠谱了。

![baseline](/pytorch_time_series_tutorial/baseline.png)

```Python
class Baseline(Model):
    def __init__(self, label_index=None):
        super(Baseline, self).__init__()
        self.label_index = label_index
    
    def forward(self, inputs):
        if self.label_index is None:
            return inputs
        else:
            return inputs[:, :, [self.label_index]]
```

实例化对象并直接在验证集和测试集上评估预测表现：

```Python
baseline = Baseline(label_index=df.columns.get_loc('T (degC)'))
baseline.compile(loss_fn=nn.MSELoss(), metric_fn=nn.L1Loss())

val_performance = {}
test_performance = {}
loss, metric = baseline.evaluate(single_step_window.val)
print(f'loss: {loss:.4f} - metric: {metric:.4f}')

val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
test_performance['Baseline'] = baseline.evaluate(single_step_window.test)
```

```
loss: 0.0128 - metric: 0.0784
```

其中评分 `metric` 使用的是 `nn.L1Loss`，即平均绝对误差（MAE）。虽然这几行代码把一些评分打印在了屏幕上，但很难让我们对模型的好坏有直观的认识。`WindowGenerator` 有画出输入、标签和预测结果的方法，但其输入和输出的时间步都只有一步，恐怕画不出什么有意思的图像。

为了方便演示，这里创建一个更宽的 `WindowGenerator` 对象，每次能切出连续 24 小时的输入和标签序列。虽然结构与 `single_step_window` 不同，但 `wide_window` 并没有改变模型的预测方式，模型依旧用一个小时的输入预测下一小时的气温。这种情况下 `time` 维就好比 `batch` 维：不同时间步的预测都是独立产生的，互不干涉。

```Python
wide_window = WindowGenerator(
    input_width=24,
    label_width=24,
    shift=1,
    label_columns=['T (degC)']
)
wide_window
```

```
Total window size: 25
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
Label column names(s): ['T (degC)']
```

扩展后的窗口能直接传给 `baseline` 模型做预测，不需要修改任何代码。这是因为初始化窗口对象时指定了输入和标签序列的时间步数相同，并且基准模型只是从输入中抽取特定几列特征作为输出，没有什么复杂的前向过程：

![last_window](/pytorch_time_series_tutorial/last_window.png)

```Python
print('Inputs shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
```

```
Inputs shape: torch.Size([32, 24, 19])
Output shape: torch.Size([32, 24, 1])
```

然后来画基准模型的预测结果，发现预测结果就是输入序列右移一个小时而已：

```Python
wide_window.plot(baseline)
```

![baseline_plot](/pytorch_time_series_tutorial/baseline_plot.png)

关于上图的一些解释：

- 蓝色的 `Inputs` 折线表示每个时间步上输入的气温。注意模型是输入了所有特征的，但这里只画出了气温而已。
- 绿色的 `Labels` 散点表示目标时间步上的真实气温。这些点显示在预测时刻而非输入时刻，这也是 `Labels` 的时间范围要比 `Inputs` 的范围向右偏移一步的原因。
- 橙色的 `Predictions` 散点是模型在输出时间步上的预测值。如果 `Predictions` 的叉叉和 `Labels` 的圆点重合，说明模型完美预测了气温。

### 线性模型

对于单步预测的任务，最简单的可训练模型就是在输入和输出之间插入线性变换层。此时输出完全由当前时间步的输入决定：

![narrow_window](/pytorch_time_series_tutorial/narrow_window.png)

一层 `torch.nn.Linear`就是一个线性模型，只会对输入的最后一维进行变换，将形如 `(batch, time, in_features)` 的数据变成 `(batch, time, out_features)` 的形状。且不同的 `batch` 和 `time` 下标都对应一个线性变换，这些变换之间互相独立：

```Python
class Linear(Model):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
    
    def forward(self, inputs):
        return self.layer(inputs)
```

```Python
linear = nn.Linear(num_features, 1)
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)
```

```
Input shape: torch.Size([32, 1, 19])
Output shape: torch.Size([32, 1, 1])
```

本教程将会训练很多模型，因此把训练流程打包成一个函数：

```Python
def compile_and_fit(model, window, max_epochs=20, patience=2):
    model.compile(
        loss_fn=nn.MSELoss(),
        metric_fn=nn.L1Loss(),
        optimizer=optim.Adam(model.parameters())
    )
    early_stopping = EarlyStopping(patience=patience)

    for t in range(max_epochs):
        loss, metric = model.train_epoch(window.train)
        val_loss, val_metric = model.evaluate(window.val)
        info = ' - '.join([
            f'[Epoch {t + 1}/{max_epochs}]',
            f'loss: {loss:.4f}',
            f'metric: {metric:.4f}',
            f'val_loss: {val_loss:.4f}',
            f'val_metric: {val_metric:.4f}'
        ])
        print(info)
        if early_stopping(val_loss):
            break
```

训练线性模型并评估其表现：

```Python
compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
test_performance['Linear'] = linear.evaluate(single_step_window.test)
```

```
[Epoch 1/20] - loss: 0.1688 - metric: 0.2229 - val_loss: 0.0176 - val_metric: 0.1021
[Epoch 2/20] - loss: 0.0151 - metric: 0.0901 - val_loss: 0.0098 - val_metric: 0.0734
[Epoch 3/20] - loss: 0.0096 - metric: 0.0718 - val_loss: 0.0089 - val_metric: 0.0701
[Epoch 4/20] - loss: 0.0092 - metric: 0.0704 - val_loss: 0.0089 - val_metric: 0.0703
[Epoch 5/20] - loss: 0.0092 - metric: 0.0704 - val_loss: 0.0088 - val_metric: 0.0690
[Epoch 6/20] - loss: 0.0092 - metric: 0.0702 - val_loss: 0.0090 - val_metric: 0.0711
[Epoch 7/20] - loss: 0.0091 - metric: 0.0700 - val_loss: 0.0088 - val_metric: 0.0696
```

跟 `baseline` 模型类似，`linear` 模型也可以直接用在宽窗口产生的批量数据上，这种用法能让模型在一串连续的时间步上给出一组互相独立的预测。此时 `time` 维的功能跟 `batch` 维类似。每个时间步上的预测互不影响。

![wide_window](/pytorch_time_series_tutorial/wide_window.png)

```Python
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)
```

```
Input shape: torch.Size([32, 24, 19])
Output shape: torch.Size([32, 24, 1])
```

注意 `wide_window` 只是方便一次性预测多个时间步和画图，训练和评估还得用 `single_step_window`，不然相当于批大小从 `batch_size` 增大为 `batch_size * input_width`。

下面画出几例 `linear` 在 `wide_window` 上的预测结果，可见大部分时刻线性模型的效果比直接用输入时刻的气温当预测更好，但在有些时刻要更差些：

```Python
wide_window.plot(linear)
```

![linear_plot](/pytorch_time_series_tutorial/linear_plot.png)

线性模型的一大好处就是易于解读，因为线性层的权重就是多变量线性回归里的系数。你可以对所有输入特征对应的权重做可视化：

```Python
x = np.arange(num_features)
_, ax = plt.subplots()
ax.bar(x, tensor_to_numpy(linear.layer.weight[0, :]))
ax.set_xticks(x)
ax.set_xticklabels(train_df.columns, rotation=90)
```

![linear_weight](/pytorch_time_series_tutorial/linear_weight.png)

有时训练出来的模型里 `T (degC)` 的权重都不是最高的（例如本图就是……），这算是随机初始化权重所带来的一个毛病。

### 密集层模型

在我们尝试多时间步输入的模型之前，有必要先来测试一下更深更强力的单时间步输入模型。

下面这个模型跟 `linear` 很像，不过在输入和输出之间多加了两层线性层和激活函数：

```Python
class Dense(Model):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
    
    def forward(self, x):
        return self.layers(x)

dense = Dense(num_features, 1)
compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
test_performance['Dense'] = dense.evaluate(single_step_window.test)
```

```
[Epoch 1/20] - loss: 0.0189 - metric: 0.0783 - val_loss: 0.0074 - val_metric: 0.0624
[Epoch 2/20] - loss: 0.0076 - metric: 0.0625 - val_loss: 0.0074 - val_metric: 0.0636
[Epoch 3/20] - loss: 0.0072 - metric: 0.0606 - val_loss: 0.0075 - val_metric: 0.0615
```

### 多步密集层模型

单时间步的模型无法获知输入在时间维度上的“上下文信息”，即模型看不到输入特征随时间的变化情况。为此模型在做预测时应该获取多个时间步的输入：

![conv_window](/pytorch_time_series_tutorial/conv_window.png)

`baseline`、`linear` 和 `dense` 模型都是单独处理每个时间步的输入，而这里要介绍的模型将会一次性接收多个时间步的输入并输出单个时间步的标签。创建窗口时注意 `shift` 参数指的是输入输出窗口末尾间的偏移量：

```Python
CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)']
)
conv_window
```

```
Total window size: 4
Input indices: [0 1 2]
Label indices: [3]
Label column names(s): ['T (degC)']
```

```Python
conv_window.plot()
ax = plt.gcf().axes[0]
ax.set_title('Given 3 hours of inputs, predict 1 hour into the future.')
```

![conv_window_plot](/pytorch_time_series_tutorial/conv_window_plot.png)

要实现这种模型，可以在 `dense` 模型前加一层 `nn.Flatten`，将形如 `(batch, time, features)` 的输入摊平成 `(batch, time * features)`，此处 `time=3`，这样一来前三个时间步的特征都会输进模型。网络最后一层再通过 `torch.Tensor.reshape` 将 `(batch, 1 * 1)` 的输出转为 `(batch, 1, 1)` 。

```Python
class nnReshape(nn.Module):
    def __init__(self, *shape):
        super(nnReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class MultiStepDense(Model):
    def __init__(self, in_features, out_features):
        super(MultiStepDense, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV_WIDTH * in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
            nnReshape(-1, 1, out_features)
        )
    
    def forward(self, x):
        return self.layers(x)
```

```Python
multi_step_dense = MultiStepDense(num_features, 1)
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
```

```
Input shape: torch.Size([32, 3, 19])
Output shape: torch.Size([32, 1, 1])
```

```Python
compile_and_fit(multi_step_dense, conv_window)
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
test_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test)
```

```
[Epoch 1/20] - loss: 0.0270 - metric: 0.0899 - val_loss: 0.0070 - val_metric: 0.0598
[Epoch 2/20] - loss: 0.0073 - metric: 0.0609 - val_loss: 0.0066 - val_metric: 0.0583
[Epoch 3/20] - loss: 0.0070 - metric: 0.0596 - val_loss: 0.0062 - val_metric: 0.0552
[Epoch 4/20] - loss: 0.0068 - metric: 0.0585 - val_loss: 0.0076 - val_metric: 0.0619
[Epoch 5/20] - loss: 0.0066 - metric: 0.0577 - val_loss: 0.0067 - val_metric: 0.0580
```

```Python
conv_window.plot(multi_step_dense)
```

![multistepdense_plot](/pytorch_time_series_tutorial/multistepdense_plot.png)

该模型的主要缺点是，要求输入和标签数据的窗口宽度必须为 `CONV_WIDTH` 和 `1`，而 `wide_window` 对象划分的数据就不能传入。

```Python
print('Input shape:', wide_window.example[0].shape)
try:
    print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
    print(f'\n{type(e).__name__}:{e}')
```

```
Input shape: torch.Size([32, 24, 19])

RuntimeError:mat1 and mat2 shapes cannot be multiplied (32x456 and 57x32)
```

下一节的卷积模型将会解决这个问题。

### 卷积神经网络

卷积层（`torch.nn.Conv1d`）同样可以用多个时间步的输入做预测。下面是跟 `multi_step_dense` **相同**的模型，不过用卷积改写了一下。改动在于：

- 用 `nn.Conv1d` 替换掉了 `nn.Flatten` 和 `nn.Linear`。不过因为 PyTorch 的一维卷积是对输入的最后一维做的，而这里希望对时间维，也就是第二维做，因此要用 `torch.Tensor.transpose` 转置一下。
- 最后不需要 `nnReshape` 层了，因为前面都保留了时间维。

```Python
class nnTranspose(nn.Module):
    def __init__(self, dim0, dim1):
        super(nnTranspose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class ConvModel(Model):
    def __init__(self, in_features, out_features):
        super(ConvModel, self).__init__()
        self.layers = nn.Sequential(
            nnTranspose(1, 2),
            nn.Conv1d(in_features, 32, CONV_WIDTH),
            nnTranspose(1, 2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    
    def forward(self, x):
        return self.layers(x)
```

在 `example` 批上检查一下模型输出的形状：

```Python
conv_model = ConvModel(num_features, 1)
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
```

```
Input shape: torch.Size([32, 3, 19])
Output shape: torch.Size([32, 1, 1])
```

在 `conv_window` 上训练并评估，其表现应该和 `multi_step_dense` 非常接近。

```Python
compile_and_fit(conv_model, conv_window)
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
test_performance['Conv'] = conv_model.evaluate(conv_window.test)
```

```
[Epoch 1/20] - loss: 0.0158 - metric: 0.0789 - val_loss: 0.0081 - val_metric: 0.0667
[Epoch 2/20] - loss: 0.0075 - metric: 0.0618 - val_loss: 0.0081 - val_metric: 0.0676
[Epoch 3/20] - loss: 0.0073 - metric: 0.0608 - val_loss: 0.0068 - val_metric: 0.0596
[Epoch 4/20] - loss: 0.0070 - metric: 0.0594 - val_loss: 0.0068 - val_metric: 0.0589
[Epoch 5/20] - loss: 0.0070 - metric: 0.0593 - val_loss: 0.0061 - val_metric: 0.0547
[Epoch 6/20] - loss: 0.0068 - metric: 0.0583 - val_loss: 0.0062 - val_metric: 0.0551
```

`conv_model` 和 `multi_step_dense` 的差异在于，`conv_model` 能在任意长度的输入上进行预测。一维卷积相当于滑动窗口版的线性层：

![wide_conv_window](/pytorch_time_series_tutorial/wide_conv_window.png)

如果你在更宽的输入上跑一下模型，会自动产生更宽的输出：

```Python
print('Wide window')
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)
```

```
Wide window
Input shape: torch.Size([32, 24, 19])
Labels shape: torch.Size([32, 24, 1])
Output shape: torch.Size([32, 22, 1])
```

可以看到输出的长度要比输入短两格，这是因为无论输入有多宽，开头 `CONV_WIDTH` 个输入总得用来“启动”第一个预测，导致预测的长度总是比输入短 `CONV_WIDTH - 1` 格。以前面的图示为例，`t=0,1,2` 时刻的输入产生 `t=3` 的预测，`t=1,2,3` 时刻的输入产生 `t=4` 的预测，后面以此类推。但 `t=1` 的预测需要 `t=-2,-1,0` 的输入，`t=2` 的预测需要 `t=-1,0,1` 的输入，而这里并没有 `t=-2,-1` 的数据，因此预测序列要比输入序列短两格。

显然 `wide_window` 产生的标签窗口并不满足这一点，所以如果想在更宽的输入上训练或画图，就需要 `WindowGenerator` 对象切出的标签窗口比输入窗口右移一步，同时开头要短两格：

```Python
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)']
)
wide_conv_window
```

```
Total window size: 27
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]
Label indices: [ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
Label column names(s): ['T (degC)']
```

```Python
print('Wide conv window')
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
```

```
Wide conv window
Input shape: torch.Size([32, 26, 19])
Labels shape: torch.Size([32, 24, 1])
Output shape: torch.Size([32, 24, 1])
```

现在终于能在更宽的窗口上画出模型的预测结果了。注意前三个输入值后才出现第一个预测值，因为每步预测都源于前三步的输入：

```Python
wide_conv_window.plot(conv_model)
```

![convmodel_plot](/pytorch_time_series_tutorial/convmodel_plot.png)

### 循环神经网络

循环神经网络（RNN）是一种特别适合处理时间序列的神经网络。RNN 会一步接一步地处理时间序列，同时用一个内部状态记录每步输入的信息。你可以在 [用 RNN 生成文本](https://www.tensorflow.org/text/tutorials/text_generation) 的教程和 [用 Keras 学循环神经网络（RNN）](https://www.tensorflow.org/guide/keras/rnn) 的指南里学到更多。

本教程用到的 RNN 层是长短期记忆（LSTM，`torch.nn.LSTM`）。将一个序列输入 LSTM 后能得到等长的序列和最后一个 cell 的状态（hidden state 和 cell state），用来做单步预测的话大致有两种思路：

1. 输出序列的最后一步已经包含了输入序列所有时间步的信息，输出序列前面每步的运算相当于是在预热（warmup）模型。这最后一步可以用作下一时刻的预测。

![lstm_1_window](/pytorch_time_series_tutorial/lstm_1_window.png)

2. 直接以输出序列作为下一时刻的预测值，相当于一次性给出了与输出序列等长的预测序列。

![lstm_many_window](/pytorch_time_series_tutorial/lstm_many_window.png)

下面用思路 2 进行演示：

```Python
class LstmModel(Model):
    def __init__(self, in_features, out_features):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(in_features, 32, batch_first=True)
        self.linear = nn.Linear(32, out_features)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.linear(output)
```

> 提示：这种用法可能使模型的表现变差，因为输出的第一步并只用到了输入第一步的信息，后面的输出里才会逐渐积累历史输入的信息。因此输出序列前几步的表现可能不比简单的 `linear` 和 `dense` 之类的模型强。

```Python
lstm_model = LstmModel(num_features, 1)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)
```

```
Input shape: torch.Size([32, 24, 19])
Output shape: torch.Size([32, 24, 1])
```

```Python
compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
test_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
```

```
[Epoch 1/20] - loss: 0.0311 - metric: 0.0900 - val_loss: 0.0063 - val_metric: 0.0552
[Epoch 2/20] - loss: 0.0063 - metric: 0.0548 - val_loss: 0.0058 - val_metric: 0.0526
[Epoch 3/20] - loss: 0.0059 - metric: 0.0529 - val_loss: 0.0056 - val_metric: 0.0517
[Epoch 4/20] - loss: 0.0057 - metric: 0.0521 - val_loss: 0.0056 - val_metric: 0.0513
[Epoch 5/20] - loss: 0.0056 - metric: 0.0514 - val_loss: 0.0055 - val_metric: 0.0508
[Epoch 6/20] - loss: 0.0055 - metric: 0.0510 - val_loss: 0.0055 - val_metric: 0.0512
[Epoch 7/20] - loss: 0.0054 - metric: 0.0505 - val_loss: 0.0057 - val_metric: 0.0519
```

```Python
wide_window.plot(lstm_model)
```

![lstmmodel_plot](/pytorch_time_series_tutorial/lstmmodel_plot.png)

### 预测表现

在本教程的数据集上，这些模型的表现应该一个比一个强：

```Python
def plot_performance(val_performance, test_performance, ylabel):
    x = np.arange(len(val_performance))
    width = 0.3
    metric_index = 1
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in test_performance.values()]

    plt.ylabel(ylabel)
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=val_performance.keys(), rotation=45)
    plt.legend()

plot_performance(
    val_performance, test_performance,
    ylabel='mean_absolute_error [T (degC), normalized]'
)
```

![single_step_performance](/pytorch_time_series_tutorial/single_step_performance.png)

```Python
for name, value in test_performance.items():
    print(f'{name:18s}: {value[1]:0.4f}')
```

```
Baseline          : 0.0853
Linear            : 0.0668
Dense             : 0.0625
Multi step dense  : 0.0578
Conv              : 0.0592
LSTM              : 0.0534
```

> 译注：卷积神经网络一节说 `Conv` 的性能应该和 `Multi step dense` 持平，但图里不是明显更差么……

### 多变量输出模型

目前为止介绍的模型都是在单个时间步上预测单个特征 `T (degC)`。

要让这些模型输出多个特征其实非常简单，只需要修改输出层的特征数，让 `WindowGenerator` 产生的标签包含所有特征即可：

```Python
single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1)
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1)

example_inputs, example_labels = wide_window.example
print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

```
Inputs shape (batch, time, features): torch.Size([32, 24, 19])
Labels shape (batch, time, features): torch.Size([32, 24, 19])
```

现在 `example_labels` 中 `features` 一维的长度和 `example_inputs` 相同了（以前是 `1`，现在是 `19`）。

#### 基准模型

缺省 `label_index` 参数即可重复所有输入特征：

```Python
baseline = Baseline()
baseline.compile(loss_fn=nn.MSELoss(), metric_fn=nn.L1Loss())
```

```Python
val_performance = {}
test_performance = {}
loss, metric = baseline.evaluate(single_step_window.val)
print(f'loss: {loss:.4f} - metric: {metric:.4f}')

val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
test_performance['Baseline'] = baseline.evaluate(single_step_window.test)
```

```
loss: 0.0892 - metric: 0.1593
```

#### 密集层模型

```Python
dense = Dense(num_features, num_features)
compile_and_fit(dense, single_step_window)
val_performance['Dense'] = dense.evaluate(single_step_window.val)
test_performance['Dense'] = dense.evaluate(single_step_window.test)
```

```
[Epoch 1/20] - loss: 0.1069 - metric: 0.1845 - val_loss: 0.0726 - val_metric: 0.1448
[Epoch 2/20] - loss: 0.0724 - metric: 0.1430 - val_loss: 0.0717 - val_metric: 0.1405
[Epoch 3/20] - loss: 0.0711 - metric: 0.1394 - val_loss: 0.0705 - val_metric: 0.1382
[Epoch 4/20] - loss: 0.0703 - metric: 0.1370 - val_loss: 0.0710 - val_metric: 0.1383
[Epoch 5/20] - loss: 0.0699 - metric: 0.1355 - val_loss: 0.0698 - val_metric: 0.1342
[Epoch 6/20] - loss: 0.0692 - metric: 0.1335 - val_loss: 0.0690 - val_metric: 0.1328
[Epoch 7/20] - loss: 0.0689 - metric: 0.1326 - val_loss: 0.0681 - val_metric: 0.1307
[Epoch 8/20] - loss: 0.0686 - metric: 0.1315 - val_loss: 0.0687 - val_metric: 0.1314
[Epoch 9/20] - loss: 0.0684 - metric: 0.1308 - val_loss: 0.0683 - val_metric: 0.1316
```

#### RNN 模型

```Python
%%time
lstm_model = LstmModel(num_features, num_features)
compile_and_fit(lstm_model, wide_window)
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
test_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
```

```
[Epoch 1/20] - loss: 0.1299 - metric: 0.2077 - val_loss: 0.0688 - val_metric: 0.1384
[Epoch 2/20] - loss: 0.0662 - metric: 0.1318 - val_loss: 0.0643 - val_metric: 0.1277
[Epoch 3/20] - loss: 0.0638 - metric: 0.1262 - val_loss: 0.0633 - val_metric: 0.1253
[Epoch 4/20] - loss: 0.0628 - metric: 0.1240 - val_loss: 0.0627 - val_metric: 0.1234
[Epoch 5/20] - loss: 0.0622 - metric: 0.1227 - val_loss: 0.0624 - val_metric: 0.1227
[Epoch 6/20] - loss: 0.0618 - metric: 0.1218 - val_loss: 0.0623 - val_metric: 0.1225
[Epoch 7/20] - loss: 0.0615 - metric: 0.1213 - val_loss: 0.0624 - val_metric: 0.1218
[Epoch 8/20] - loss: 0.0613 - metric: 0.1209 - val_loss: 0.0625 - val_metric: 0.1217
CPU times: total: 12.7 s
Wall time: 2min 23s
```

#### 高级方法：残差连接

前面的基准模型利用了数据集这样一个特性：序列中相邻两步间的差异并不是很大。而其它需要训练的模型首先会随机初始化权重参数，然后才在训练中逐渐学到输出值相比前一步只变化了一点的事实。虽然你可以通过微调初始化方法来解决这一问题，但更简单的做法是把这种关系纳入模型结构中。

不直接预测下一步的数值而预测下一步的改变量，在时间序列建模中是很常见的策略。类似的，深度学习中的 [残差神经网络](https://arxiv.org/abs/1512.03385)（ResNets）就是一种将每层输出加到模型累积结果里的结构。

改变很小这一特性，就是这样被利用起来的。

![residual](/pytorch_time_series_tutorial/residual.png)

本质上讲，该结构相当于把模型初始化到跟 `Baseline` 一样的状态。对于当下的预测任务，该结构能使模型收敛更快，稍稍提高模型的性能。除此之外，该结构还能结合本教程提到的其它模型使用。

这里结合 LSTM 进行演示，注意用到了 `torch.nn.init.zeros_` 将 LSTM 最后一层的权重置零，以确保训练刚开始时预测出的改变量足够小，并且不会盖过残差连接的效果。因为置零仅对最后一层进行，所以不必担心梯度会出现 symmetry-breaking 的问题。

```Python
class ResidualWrapper(Model):
    def __init__(self, model):
        super(ResidualWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        dx = self.model(x)
        return x + dx
```

```Python
%%time
lstm_model = LstmModel(num_features, num_features)
nn.init.zeros_(lstm_model.linear.weight) # 直接修改参数的话需要用no_grad包裹.
residual_lstm = ResidualWrapper(lstm_model)

compile_and_fit(residual_lstm, wide_window)
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
test_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test)
```

```
[Epoch 1/20] - loss: 0.0658 - metric: 0.1238 - val_loss: 0.0631 - val_metric: 0.1193
[Epoch 2/20] - loss: 0.0620 - metric: 0.1181 - val_loss: 0.0624 - val_metric: 0.1182
[Epoch 3/20] - loss: 0.0609 - metric: 0.1169 - val_loss: 0.0622 - val_metric: 0.1178
[Epoch 4/20] - loss: 0.0603 - metric: 0.1163 - val_loss: 0.0619 - val_metric: 0.1174
[Epoch 5/20] - loss: 0.0598 - metric: 0.1159 - val_loss: 0.0619 - val_metric: 0.1173
[Epoch 6/20] - loss: 0.0593 - metric: 0.1155 - val_loss: 0.0623 - val_metric: 0.1175
CPU times: total: 15.4 s
Wall time: 1min 55s
```

#### 预测表现

下面是这些多变量输出模型的总体表现：

```Python
plot_performance(
    val_performance, test_performance,
    ylabel='MAE (average over all outputs)'
)
```

![multi_output_performance](/pytorch_time_series_tutorial/multi_output_performance.png)

```Python
for name, value in test_performance.items():
    print(f'{name:15s}: {value[1]:0.4f}')
```

```
Baseline       : 0.1633
Dense          : 0.1332
LSTM           : 0.1237
Residual LSTM  : 0.1188
```

以上评分取模型所有输出的平均值。

## 多步模型

前面几节的单变量输出和多变量输出模型都只能预测一个时间步，即一小时后。本节将介绍如何将这些模型扩展成能预测多个时间步的版本。

在做多步预测时，模型需要学习预测未来一段时间范围的值。因此多步模型不像单步模型那样只能给出未来一个时刻的预测，而是会输出一段预测序列。为此大致有两种方法：

- 单发预测（single-shot），即一次性预测整段时间序列。
- 自回归预测（autoregressive），即模型做单步预测后，再把这步结果作为输入喂给模型，得到下下步的预测，以此类推。

本节的模型将在输出时间步上预测**所有特征**（即多变量输出）。

对于多步模型，训练数据同样由每小时的样本组成。不同的是模型将学习用过去 24 小时的输入预测未来 24 小时的特征。

下面的 `Window` 对象会从数据集中生成我们需要的切片：

```Python
OUT_STEPS = 24
multi_window = WindowGenerator(
    input_width=24,
    label_width=OUT_STEPS,
    shift=OUT_STEPS
)
multi_window.plot()
multi_window
```

```
Total window size: 48
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Label indices: [24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47]
Label column names(s): ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'Wx', 'Wy', 'max Wx', 'max Wy', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
```

![multi_window_plot](/pytorch_time_series_tutorial/multi_window_plot.png)

### 基准模型

对该任务来说一个简单的基准模型就是把输入最后一步的数值重复 `OUT_STEPS` 次：

![multistep_last](/pytorch_time_series_tutorial/multistep_last.png)

```Python
class MultiStepLastBaseline(Model):
    def __init__(self):
        super(MultiStepLastBaseline, self).__init__()
    
    def forward(self, x):
        return x[:, -1:, :].tile(1, OUT_STEPS, 1)

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss_fn=nn.MSELoss(), metric_fn=nn.L1Loss())

multi_val_performance = {}
multi_test_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_test_performance['Last'] = last_baseline.evaluate(multi_window.test)
multi_window.plot(last_baseline)
```

```
loss: 0.6286 - metric: 0.5007
```

![multistepbaseline_plot](/pytorch_time_series_tutorial/multistepbaseline_plot.png)

因为预测任务是用过去 24 小时预测未来 24 小时，两段时间正好都是一天的长度，所以另一个简单的方法是假设明天和今天的时间序列差不多，直接重复今天的序列作为预测：

![multistep_repeat](/pytorch_time_series_tutorial/multistep_repeat.png)

```Python
class RepeatBaseline(Model):
    def __init__(self):
        super(RepeatBaseline, self).__init__()
    
    def forward(self, x):
        return x

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss_fn=nn.MSELoss(), metric_fn=nn.L1Loss())
loss, metric = repeat_baseline.evaluate(multi_window.val)
print(f'loss: {loss:.4f} - metric: {metric:.4f}')

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_test_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test)
multi_window.plot(repeat_baseline)
```

```
loss: 0.4271 - metric: 0.3959
```

![repeatbaseline_plot](/pytorch_time_series_tutorial/repeatbaseline_plot.png)

### 单发模型

一个更高级点的方法是使用“单发”模型，即模型会一次性预测未来的整段序列。

将 `torch.nn.Linear` 的输出特征数设为 `OUT_STEPS * features` 即可轻松实现这种模型，只是模型的最后一层需要将输出变形成 `(batch, OUT_STEPS, features)` 的形状。

#### 线性模型

一个用输入序列最后一步做预测的简单的线性模型表现会比前面两种基准模型要好，但好也好不到哪里去。该模型会把单步输入线性投影成 `OUT_STEPS` 步的预测结果，因此只能捕捉到序列行为里低维的部分，很可能主要靠时间在一天或一年中的位置来做预测。

![multistep_dense](/pytorch_time_series_tutorial/multistep_dense.png)

```Python
class MultiLinearModel(Model):
    def __init__(self, in_features, out_features):
        super(MultiLinearModel, self).__init__()
        self.layers = nn.Sequential(
            # Shape => [batch, 1, OUT_STEPS * features]
            nn.Linear(in_features, OUT_STEPS * out_features),
            # Shape => [batch, OUT_STEPS, features]
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.zeros_(self.layers[0].weight)
    
    def forward(self, x):
        # Shape [batch, time, features] => [batch, 1, features]
        return self.layers(x[:, -1:, :])

multi_linear_model = MultiLinearModel(num_features, num_features)
compile_and_fit(multi_linear_model, multi_window)

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_test_performance['Linear'] = multi_linear_model.evaluate(multi_window.test)
multi_window.plot(multi_linear_model)
```

```
[Epoch 1/20] - loss: 0.3296 - metric: 0.3990 - val_loss: 0.2585 - val_metric: 0.3232
[Epoch 2/20] - loss: 0.2570 - metric: 0.3127 - val_loss: 0.2558 - val_metric: 0.3059
[Epoch 3/20] - loss: 0.2562 - metric: 0.3074 - val_loss: 0.2559 - val_metric: 0.3055
[Epoch 4/20] - loss: 0.2561 - metric: 0.3072 - val_loss: 0.2550 - val_metric: 0.3049
[Epoch 5/20] - loss: 0.2560 - metric: 0.3071 - val_loss: 0.2558 - val_metric: 0.3052
[Epoch 6/20] - loss: 0.2560 - metric: 0.3071 - val_loss: 0.2555 - val_metric: 0.3050
```

![multilinearmodel_plot](/pytorch_time_series_tutorial/multilinearmodel_plot.png)

#### 密集层模型

在线性模型的输入输出之间再加一层 `torch.nn.Linear` 能增强其表现，不过说到底还是只用了输入最后一步的信息。

```Python
class MultiDenseModel(Model):
    def __init__(self, in_features, out_features):
        super(MultiDenseModel, self).__init__()
        self.dense = nn.Sequential(
            # Shape => [batch, 1, 512]
            nn.Linear(in_features, 512),
            nn.ReLU(),
            # Shape => [batch, 1, OUT_STEPS * features]
            nn.Linear(512, OUT_STEPS * out_features),
            # Shape => [batch, OUT_STEPS, features]
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.zeros_(self.dense[2].weight)
    
    def forward(self, x):
        # Shape [batch, time, features] => [batch, 1, features]
        return self.dense(x[:, -1:, :])

multi_dense_model = MultiDenseModel(num_features, num_features)
compile_and_fit(multi_dense_model, multi_window)

multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_test_performance['Dense'] = multi_dense_model.evaluate(multi_window.test)
multi_window.plot(multi_dense_model)
```

```
[Epoch 1/20] - loss: 0.2346 - metric: 0.2957 - val_loss: 0.2249 - val_metric: 0.2854
[Epoch 2/20] - loss: 0.2204 - metric: 0.2826 - val_loss: 0.2217 - val_metric: 0.2845
[Epoch 3/20] - loss: 0.2169 - metric: 0.2798 - val_loss: 0.2208 - val_metric: 0.2835
[Epoch 4/20] - loss: 0.2146 - metric: 0.2781 - val_loss: 0.2179 - val_metric: 0.2806
[Epoch 5/20] - loss: 0.2130 - metric: 0.2771 - val_loss: 0.2194 - val_metric: 0.2824
[Epoch 6/20] - loss: 0.2118 - metric: 0.2763 - val_loss: 0.2194 - val_metric: 0.2809
```

![multidensemodel_plot](/pytorch_time_series_tutorial/multidensemodel_plot.png)

#### CNN

卷积模型会用固定宽度的历史输入做预测，因为能看到输入是如何随时间变化的，所以表现可能比密集层模型要好点：

![multistep_conv](/pytorch_time_series_tutorial/multistep_conv.png)

```Python
CONV_WIDTH = 3
class MultiConvModel(Model):
    def __init__(self, in_features, out_features):
        super(MultiConvModel, self).__init__()
        self.layers = nn.Sequential(
            # Shape => [batch, 1, 256]
            nnTranspose(1, 2),
            nn.Conv1d(in_features, 256, CONV_WIDTH),
            nnTranspose(1, 2),
            nn.ReLU(),
            # Shape => [batch, 1, OUTSTEPS * features]
            nn.Linear(256, OUT_STEPS * out_features),
            # Shape => [batch, OUTSTEPS, features]
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.zeros_(self.layers[4].weight)

    def forward(self, x):
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        return self.layers(x[:, -CONV_WIDTH:, :])

multi_conv_model = MultiConvModel(num_features, num_features)
compile_and_fit(multi_conv_model, multi_window)

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_test_performance['Conv'] = multi_conv_model.evaluate(multi_window.test)
multi_window.plot(multi_conv_model)
```

```
[Epoch 1/20] - loss: 0.2360 - metric: 0.3019 - val_loss: 0.2223 - val_metric: 0.2874
[Epoch 2/20] - loss: 0.2188 - metric: 0.2853 - val_loss: 0.2193 - val_metric: 0.2854
[Epoch 3/20] - loss: 0.2149 - metric: 0.2819 - val_loss: 0.2174 - val_metric: 0.2838
[Epoch 4/20] - loss: 0.2122 - metric: 0.2797 - val_loss: 0.2176 - val_metric: 0.2846
[Epoch 5/20] - loss: 0.2101 - metric: 0.2781 - val_loss: 0.2169 - val_metric: 0.2822
[Epoch 6/20] - loss: 0.2083 - metric: 0.2765 - val_loss: 0.2164 - val_metric: 0.2816
[Epoch 7/20] - loss: 0.2069 - metric: 0.2755 - val_loss: 0.2124 - val_metric: 0.2782
[Epoch 8/20] - loss: 0.2058 - metric: 0.2744 - val_loss: 0.2161 - val_metric: 0.2825
[Epoch 9/20] - loss: 0.2049 - metric: 0.2738 - val_loss: 0.2154 - val_metric: 0.2806
```

![multiconvmodel_plot](/pytorch_time_series_tutorial/multiconvmodel_plot.png)

#### RNN

RNN 模型能够学习用长期的历史数据做预测。下面这个模型会用内部状态积累 24 小时历史数据里的信息，然后一次性预报接下来 24 小时。

我们在这种单发预测里只需要 LSTM 输出的最后一个时间步：

![multistep_lstm](/pytorch_time_series_tutorial/multistep_lstm.png)

```Python
class MultiLstmModel(Model):
    def __init__(self, in_features, out_features):
        super(MultiLstmModel, self).__init__()
        self.lstm = nn.LSTM(in_features, 32, batch_first=True)
        self.linear = nn.Sequential(
            # Shape => [batch, 1, OUT_STEPS * features]
            nn.Linear(32, OUT_STEPS * out_features),
            # Shape => [batch, OUT_STEPS, features]
            nnReshape(-1, OUT_STEPS, out_features)
        )
        nn.init.zeros_(self.linear[0].weight)
    
    def forward(self, x):
        # Shape [batch, time, features] => [batch, time, 32]
        output, _ = self.lstm(x)
        # Shape => [batch, 1, 32]
        return self.linear(output[:, -1:, :])

multi_lstm_model = MultiLstmModel(num_features, num_features)
compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_test_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test)
multi_window.plot(multi_lstm_model)
```

```
[Epoch 1/20] - loss: 0.2907 - metric: 0.3612 - val_loss: 0.2290 - val_metric: 0.3052
[Epoch 2/20] - loss: 0.2167 - metric: 0.2937 - val_loss: 0.2203 - val_metric: 0.2947
[Epoch 3/20] - loss: 0.2095 - metric: 0.2853 - val_loss: 0.2178 - val_metric: 0.2913
[Epoch 4/20] - loss: 0.2054 - metric: 0.2809 - val_loss: 0.2188 - val_metric: 0.2896
[Epoch 5/20] - loss: 0.2027 - metric: 0.2780 - val_loss: 0.2133 - val_metric: 0.2853
[Epoch 6/20] - loss: 0.2006 - metric: 0.2761 - val_loss: 0.2124 - val_metric: 0.2832
[Epoch 7/20] - loss: 0.1989 - metric: 0.2744 - val_loss: 0.2142 - val_metric: 0.2844
[Epoch 8/20] - loss: 0.1973 - metric: 0.2730 - val_loss: 0.2133 - val_metric: 0.2823
```

![multilstmmodel_plot](/pytorch_time_series_tutorial/multilstmmodel_plot.png)

### 高级方法：自回归模型

上面几个模型都是一次性预报一整段序列，但有些时候让模型把预测分解到每一步上会更有帮助。然后，模型在每一步上的输出都会回馈给模型自身，下一步的预测可以基于前一步的预测来做，就像经典的 [用循环神经网络生成序列](https://arxiv.org/abs/1308.0850) 一样。

这类模型最显著的优点是，能够产生任意时间长度的预测。

你当然可以用教程前面训练好的单步模型做自回归的循环（输出单步再输回去），但这里我们想搭一个直接把自回归纳入训练过程的模型。

![multistep_autoregressive](/pytorch_time_series_tutorial/multistep_autoregressive.png)

#### RNN

本教程只演示自回归的 RNN 模型，但其套路可以沿用于其它单步输出的模型。该模型的基础和之前的单步 LSTM 模型一样：`torch.nn.LSTM` 输出序列的最后一步被 `torch.nn.Linear` 变换成了预测值。

```Python
class FeedBack(Model):
    def __init__(self, num_features, out_steps):
        super(FeedBack, self).__init__()
        self.out_steps = out_steps
        self.lstm = nn.LSTM(num_features, 32, batch_first=True)
        self.linear = nn.Linear(32, num_features)
    
    def next(self, x, hc=None):
        # Shape [batch, time, features] => [batch, 1, features]
        output, hc = self.lstm(x, hc)
        prediction = self.linear(output[:, -1:, :])
        return prediction, hc
    
    def forward(self, x):
        predictions = []
        prediction, hc = self.next(x)
        predictions.append(prediction)
        
        for i in range(1, self.out_steps):
            prediction, hc = self.next(prediction, hc)
            predictions.append(prediction)
        # Shape => [batch, out_steps, features]
        predictions = torch.cat(predictions, dim=1)
        
        return predictions
```

最开始我们将形如 `(batch, time, features)` 的输入序列传入 `next` 方法中，得到形如 `(batch, 1, features)` 的单步预测值 `prediction` 和 LSTM 网络中最后一个 cell 的状态 `hc`。本来应该是把输入序列和 `prediction` 连起来作为新的输入序列传入 `next` 方法，得到下下步的预测。但得益于 `hc` 记录了之前序列里的信息，我们可以只把 `prediction` 和 `hc` 传入 `next` 方法，得到下下步的预测。如此循环操作 `out_steps` 步，再将这些结果用 `torch.cat` 连起来，便得到了形如 `(batch, out_steps, features)` 的输出序列。

```Python
feedback = FeedBack(num_features, OUT_STEPS)
compile_and_fit(feedback, multi_window)

multi_val_performance['AR LSTM'] = feedback.evaluate(multi_window.val)
multi_test_performance['AR LSTM'] = feedback.evaluate(multi_window.test)
multi_window.plot(feedback)
```

```
[Epoch 1/20] - loss: 0.3593 - metric: 0.4166 - val_loss: 0.2748 - val_metric: 0.3589
[Epoch 2/20] - loss: 0.2522 - metric: 0.3367 - val_loss: 0.2539 - val_metric: 0.3354
[Epoch 3/20] - loss: 0.2369 - metric: 0.3208 - val_loss: 0.2415 - val_metric: 0.3198
[Epoch 4/20] - loss: 0.2264 - metric: 0.3079 - val_loss: 0.2348 - val_metric: 0.3141
[Epoch 5/20] - loss: 0.2207 - metric: 0.3013 - val_loss: 0.2339 - val_metric: 0.3080
[Epoch 6/20] - loss: 0.2172 - metric: 0.2975 - val_loss: 0.2287 - val_metric: 0.3036
[Epoch 7/20] - loss: 0.2136 - metric: 0.2941 - val_loss: 0.2268 - val_metric: 0.3012
[Epoch 8/20] - loss: 0.2114 - metric: 0.2917 - val_loss: 0.2270 - val_metric: 0.3006
[Epoch 9/20] - loss: 0.2086 - metric: 0.2889 - val_loss: 0.2231 - val_metric: 0.2963
[Epoch 10/20] - loss: 0.2065 - metric: 0.2873 - val_loss: 0.2260 - val_metric: 0.3001
[Epoch 11/20] - loss: 0.2040 - metric: 0.2850 - val_loss: 0.2286 - val_metric: 0.3018
```

![feedback_plot](/pytorch_time_series_tutorial/feedback_plot.png)

### 预测表现

在多步预测的问题上，显然收益随模型复杂度的升高而递减：

```Python
plot_performance(
    multi_val_performance, multi_test_performance,
    ylabel='MAE (average over all times and outputs)'
)
```

![multi_step_performance](/pytorch_time_series_tutorial/multi_step_performance.png)

本教程多变量输出模型一节的评分图是对所有输出特征做平均后画出来的，这里的评分也类似，不过进一步在输出的时间步上也做了平均。

```Python
for name, value in multi_test_performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')
```

```
Last    : 0.5156
Repeat  : 0.3774
Linear  : 0.2983
Dense   : 0.2756
Conv    : 0.2744
LSTM    : 0.2730
AR LSTM : 0.2922
```

将密集层模型升级成卷积模型和循环模型后获得的收益仅有微小的几个百分点，自回归模型甚至表现变得更差了。因此，这些复杂的模型可能并不适用于**该问题**，但实际应用中模型是好是坏只有动手试一试才会知道，说不定这些模型有助于解决**你的问题**呢。

## 下一步

本教程简单介绍了如何使用 PyTorch 预测时间序列。更多相关教程请参阅：

- [《机器学习实战：基于 Scikit-Learn、Keras 和 TensorFlow》](https://book.douban.com/subject/35218199/) 第二版的 15 章。
- [《Python 深度学习》](https://book.douban.com/subject/30293801/) 的第 6 章。
- [Udacity 的 Intro to TensorFlow for Deep Learning](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) 的第 8 课，包含 [练习题](https://github.com/tensorflow/examples/tree/master/courses/udacity_intro_to_tensorflow_for_deep_learning)。

此外，虽然本教程只关注 PyTorch 内置的功能，但你也可以用来实现任何 [经典的时间序列模型](https://otexts.com/fpp3/)。