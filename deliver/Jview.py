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
    xx, yy = np.meshgrid(np.linspace(wmin[0], wmax[0],100), np.linspace(wmin[1], wmax[1],100))
    w_grid = np.c_[xx.ravel(), yy.ravel()]
    #print(xx.shape)
    #print(w_grid.shape)
    #print(X_bias.shape)

    # Cálculo do J(w) para todos os w da matriz de parâmetros

    J_grid = compute_cost(X_bias, y, w_grid.T)

    # Plotagem de J na matriz de parâmetros
    J_grid = J_grid.reshape(xx.shape)
    plt.pcolormesh(xx, yy, J_grid, cmap=plt.cm.Paired)

    # Plotagem dos pontos da sequência dos parâmetros durante o processo do gradiente descendente

    plt.scatter(w_history[:,0],w_history[:,1])
    plt.scatter(w_opt[0],w_opt[1],marker='x', c='b') # Solução analítica
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
    xx, yy = np.meshgrid(np.linspace(wmin[0], wmax[0],100), np.linspace(wmin[1], wmax[1],100))
    w_grid = np.c_[xx.ravel(), yy.ravel()]
    #print(xx.shape)
    #print(w_grid.shape)
    #print(X_bias.shape)

    # Cálculo do J(w) para todos os w da matriz de parâmetros

    J_grid = compute_cost(X_bias, y, w_grid.T)

    # Plotagem de J na matriz de parâmetros
    J_grid = J_grid.reshape(xx.shape)

    fig = plt.figure(figsize=(35,17.75))
    ax = fig.add_subplot(111, projection='3d')

    #plota a superfcie 3D
    ax.plot_surface(xx, yy, J_grid, rstride=3, cstride=3,  alpha=0.1, cmap=cm.coolwarm)
    #Plota o historicos do vetor W
    ax.scatter(w_history[:,0],w_history[:,1],c='r',marker = 'o',s = 80)
    ax.scatter(w_opt[0],w_opt[1],marker='x', c='b',s = 80)
    ax.set_xlabel(r'$w_0$',fontsize = 35)
    ax.set_ylabel(r'$w_1$',fontsize = 35)
    ax.set_zlabel('Custo (J)',fontsize = 35);
    