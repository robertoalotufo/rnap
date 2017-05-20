import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X_b, y, wT):
    '''
    Compute cost for linear regression
    (X,y): amostras rotuladas X(n_samples,2) e y(n_samples,)
    wT: vetor coluna de parâmetros (já transposto)
       aceita tanto shape (2,1) Para um caso como (2,n_history) para n_history casos
    '''
    n_samples = y.size  # número de amostras
    e = X_b.dot(wT) - y
    J = (1./n_samples) * (e * e).sum(axis=0)
    return J

# Cálculo dos limites para gerar o espaço de parâmetros
def Jview(X_bias, y, w_history, w_opt):
    wmin = w_history.min(axis=0)
    wmax = w_history.max(axis=0)
    D = wmax - wmin
    wmin -= D
    wmax += D
    #print('wmin:', wmin)
    #print('wmax:', wmax)

    # Cálculo da matriz bidimensional de parâmetros
    ww0, ww1 = np.meshgrid(np.linspace(wmin[0], wmax[0],100), 
                         np.linspace(wmin[1], wmax[1],100))
    w_grid = np.c_[ww0.ravel(), ww1.ravel()]
    #print(xx.shape)
    #print(w_grid.shape)
    #print(X_bias.shape)

    # Cálculo do J(w) para todos os w da matriz de parâmetros

    J_grid = compute_cost(X_bias, y, w_grid.T)

    # Plotagem de J na matriz de parâmetros
    J_grid = J_grid.reshape(ww0.shape)
    plt.pcolormesh(ww0, ww1, J_grid, cmap=plt.cm.coolwarm)
    plt.contour(ww0, ww1, J_grid, 20)

    # Plotagem dos pontos da sequência dos parâmetros durante o processo do gradiente descendente

    plt.scatter(w_history[:,0],w_history[:,1],c='r',marker = 'o')
    plt.scatter(w_opt[0],w_opt[1],marker='x', c='w') # Solução analítica
    plt.title('Visualização do treinamento de w na função de Custo J')
    plt.show()
    
def Jview3D(X_bias, y, w_history, w_opt):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    wmin = w_history.min(axis=0)
    wmax = w_history.max(axis=0)
    D = wmax - wmin
    wmin -= D
    wmax += D
    #print('wmin:', wmin)
    #print('wmax:', wmax)

    # Cálculo da matriz bidimensional de parâmetros
    ww0, ww1 = np.meshgrid(np.linspace(wmin[0], wmax[0],100), 
                         np.linspace(wmin[1], wmax[1],100))
    w_grid = np.c_[ww0.ravel(), ww1.ravel()]

    # Cálculo do J(w) para todos os w da matriz de parâmetros
    J_grid = compute_cost(X_bias, y, w_grid.T)

    # Plotagem de J na matriz de parâmetros
    J_grid = J_grid.reshape(ww0.shape)

    fig = plt.figure(figsize=(35,17.75))
    ax = fig.add_subplot(111, projection='3d')

    #plota a superfcie 3D
    ax.plot_surface(ww0, ww1, J_grid, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(ww0,ww1,J_grid,20)
    
    #Plota o historicos do vetor W
    J_history = compute_cost(X_bias, y, w_history.T)
    ax.scatter(w_history[:,0],w_history[:,1],J_history,c='r',marker = 'o',s = 80)
    
    # Marca o ponto ótimo pela solução analítica
    ax.scatter(w_opt[0],w_opt[1],marker='x', c='b',s = 80)
    
    ax.set_xlabel(r'$w_0$',fontsize = 35)
    ax.set_ylabel(r'$w_1$',fontsize = 35)
    ax.set_zlabel('Custo (J)',fontsize = 35);

def softmax(Z):
    # computes softmax for all samples, normalize among classes (columns)
    # input Z: scores; shape: samples rows x classes columns
    # output S: same shape of input
    EZ = np.exp(Z)
    S = EZ / EZ.sum(axis=1,keepdims=True) # normaliza nas classes - colunas
    return S

def SMpredict(X,WT):
    S = softmax(X.dot(WT))
    # escolhe a maior probabilidade entre as classes
    Y_hat = np.argmax(S,axis=1)
    return Y_hat

def FSView(X_bias,Y,WT):
    h = .02  # step size in the mesh
    folga = 0.1
    # Calcula a grade para o espaço de atributos
    X_c = X_bias[:,1:]
    x_min, x_max = X_c.min(axis=0) - folga, X_c.max(axis=0) + folga
    xx, yy = np.meshgrid(np.arange(x_min[0], x_max[0], h), np.arange(x_min[1], x_max[1], h))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid = np.hstack([np.ones((X_grid.shape[0],1)),X_grid]) # incluído X00 como 1 para gerar bias no W

    # Faz a predição para todas as amostras do espaço de atributos
    Z = SMpredict(X_grid, WT)

    # Mostra o resultado da predição (0, 1 ou 2) no gráfico
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Mostra os pontos das amostras de treinamento
    colors = np.array(['r','y','b'])
    plt.scatter(X_bias[:, 1], X_bias[:, 2], c=colors[Y], edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
    
def FSView_keras(Xc,Y,model):
    h = .02  # step size in the mesh
    folga = 0.1
    # Calcula a grade para o espaço de atributos

    x_min, x_max = Xc.min(axis=0) - folga, Xc.max(axis=0) + folga
    xx, yy = np.meshgrid(np.arange(x_min[0], x_max[0], h), np.arange(x_min[1], x_max[1], h))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    #X_grid = np.hstack([np.ones((X_grid.shape[0],1)),X_grid]) # incluído X00 como 1 para gerar bias no W

    # Faz a predição para todas as amostras do espaço de atributos
    Z = model.predict_classes(X_grid)

    # Mostra o resultado da predição (0, 1 ou 2) no gráfico
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(6, 5))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Mostra os pontos das amostras de treinamento
    colors = np.array(['r','y','b'])
    plt.scatter(Xc[:, 0], Xc[:, 1], c=colors[Y], edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()