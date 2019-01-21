import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def std_PCA(score):
    scaler = MinMaxScaler()
    pca = PCA(score)
    pipeline = Pipeline([('scaler', scaler),
                         ('pca', pca)])

    return pipeline

def Pca(x, score, x_start):
    pca = std_PCA(score)
    pca.fit(x)
    Z = pca.transform(x) # transform就会执行降维操作
    # print("降维后的矩阵:\n{}".format(Z))
    dataframe = pd.DataFrame(Z)
    filename = '降维后的矩阵' + '.csv'
    dataframe.to_csv(filename, sep=',')
    score = pca.score(x) # 所有样本的对数似然函数平均值(特定条件下经验风险最小化)
    print('fit score:{}'.format(score))
    '''
    See. "Pattern Recognition and Machine Learning" by C. Bishop, 12.2.1 p. 574
    or http://www.miketipping.com/papers/met-mppca.pdf
    '''
    restore = pca.inverse_transform(Z)
    # print("回复后的矩阵:\n{}".format(restore))

    pca = pca.named_steps['pca']
    Ureduce = pca.components_ # 得到降维用的Ureduce，即主成分分解的特征向量
    dataframe = pd.DataFrame(Ureduce)
    filename = '特征向量' + '.csv'
    dataframe.to_csv(filename, sep=',')
    # print(pca.explained_variance_)
    # 由每个所选组件解释的方差量。等于X的协方差矩阵的n_components最大特征值。
    explained_variance_data = pca.explained_variance_ratio_  # 所保留的n个成分各自的方差百分比
    cum_explained_variance_data = np.cumsum(explained_variance_data)  # 方差比累积和
    # print('所保留的n个成分各自的方差百分比:{}'.format(explained_variance_data))
    print('交叉验证评估分数 :' + str(np.mean(cross_val_score(pca, x, cv=5))))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # 画图
    number = len(explained_variance_data)
    sum = cum_explained_variance_data[-1]
    axes[0].grid(True)
    axes[0].axis([0, number, 0, 1])  # plt.axis([xmin, xmax, ymin, ymax])
    axes[0].bar(range(number), explained_variance_data, align="edge", width=1.0, alpha=0.5,
            label='individual explained variance')
    axes[0].step(range(number), cum_explained_variance_data, where='post',
             label='cumulative explained variance')

    list_temp = []
    for i in range(number+1):
        list_temp.append(sum)
    axes[0].plot(range(number+1), list_temp, '--')
    axes[0].text(0, sum, sum)

    axes[0].set_ylabel('Explained variance ratio')
    axes[0].set_xlabel('Principal components')
    axes[0].legend(loc='best')

    cmap = plt.get_cmap("tab20c")
    color = np.array(range(0, number, 1))
    color = cmap(color)
    axes[1].pie(pca.explained_variance_, radius=1, autopct='%1.1f%%', colors=color,
                wedgeprops=dict(width=0.3, edgecolor='w'))
    axes[1].set_xlabel('Explained variance ratio')
    plt.show()

    # specify columns to plot
    groups = range(Z.shape[1])
    i = 1
    # plot each column
    plt.figure()
    pca_x = np.arange(x_start, x_start+len(Z[:, 0]))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(pca_x, Z[:, group])
        plt.title('%f'%explained_variance_data[group], y=0.5, loc='right')
        i += 1
    plt.xticks(rotation=30)  # 设置坐标轴刻度 & 旋转
    plt.show()

    return Z, Ureduce

if __name__ == '__main__':

    data = pd.read_csv('Nowdata.csv')
    data.drop(columns='LABELS', inplace=True)
    # 删掉'LABELS'那一列
    data.dropna(axis=1, how='any', inplace=True)
    '''
    axis：0-行操作（默认），1-列操作 
    how：any-只要有空值就删除（默认），all-全部为空值才删除 
    inplace：False-返回新的数据集（默认），True-在愿数据集上操作
    '''
    # x = data.fillna(0)  空位补0
    filename = '预处理后数据' + '.csv'
    data.to_csv(filename, sep=',')

    Z, Ureduce = Pca(data, 0.9, x_start = 1981)
    import lstm
    lstm.Lstm(Z[:,0], x_start = 1981, predict_num=10, look_back = 2)
