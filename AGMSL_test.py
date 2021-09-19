#%%
import pandas as pd
import numpy as np
from math import sqrt
from numpy.linalg import inv, norm, eigh, slogdet
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score,  coverage_error, label_ranking_average_precision_score, label_ranking_loss

eps = np.finfo(np.float).eps

#%%
def soft_threshold(Z, a):
    G = np.where(Z-a > 0, Z-a, 0) - np.where(-Z-a > 0, -Z-a, 0)
    return G


def gradient_f(W, X, Y, L, O, gam, beta):
    return X@X.T@W -X@Y.T + gam*X@L@X.T@W + beta*W@O


def objectvie_f(W, X, Y, L, O, gam, beta):
    p1 = 0.5*norm(Y - W.T@X, 'fro')**2
    p2 = 0.5*gam*np.trace(W.T@X@L@X.T@W)
    p3 = beta/2*np.trace(W@O@W.T)
    return p1 + p2 + p3


def objective_q(W, W0, X, Y, L, O, lf, gam, beta):
    p1 = objectvie_f(W0, X, Y, L, O, gam, beta)
    
    df = gradient_f(W0, X, Y, L, O, gam, beta)
    p2 = np.trace((W - W0).T@df)
    
    p3 = 1/(2*lf)*norm(W - W0, 'fro')**2
    
    return p1 + p2 + p3


def obj_fun(Ws, Xs, distXs, Y, L, S, O, param):
    gam = param['gam']
    mu = param['mu']
    beta = param['beta']
    lam1 = param['lam1']
    lam2 = param['lam2']
    
    l = O.shape[0]
    num_view = len(Ws)
#    rho = param['rho']
    p1 = np.sum([objectvie_f(Ws[i], Xs[i], Y, L, O, gam, beta) for i in range(num_view)])
    p2 = np.sum([lam1*norm(Ws[i], 1) for i in range(num_view)])
    sign, O_log_det = slogdet(O)
    p3 = lam2*norm(O, 1) - l*sign*O_log_det
    
    p4 = norm(S, 'fro')**2
    for i in range(num_view):
        w = 0.5/np.sqrt(np.sum(distXs[i]*S))
        p4 += mu*np.sum(w*distXs[i]*S)
    
    return p1 + p2 + p3 + p4


def acc_proximal_gradient(W_old, X, Y, L, O, gam, beta, lam1, max_iter = 10):
    d, n = X.shape
    t0 = 0
    t1 = 1
    lf = 1
    
    W = W_old.copy()
    
    for it_w in range(max_iter):
        a = (t0 - 1)/t1
        G = (1+a)*W - a*W_old
        
        df = gradient_f(G, X, Y, L, O, gam, beta)
        while 1:
            G1 = soft_threshold(G - lf*df, lam1*lf)
            objF = objectvie_f(G1, X, Y, L, O, gam, beta)
            objQ = objective_q(G1, G, X, Y, L, O, lf, gam, beta)
            if objF > objQ:
                lf = lf/2
#                print(lf)
            else:
                break
        
        W_old = W
        W = G1
        
        t0, t1 = t1, (1 + sqrt(1 + 4*t0**2))/2
    
    diff_W = norm(W - W_old, 'fro')
    
    return W, diff_W


def covsel_admm(O, Ws, rho, beta, lam2, max_iter = 10):
    l = O.shape[0]
    
    A = np.zeros((l, l))
    for i in range(len(Ws)):
        A += Ws[i].T@Ws[i]/l
#    A = np.cov(W, rowvar = False)
    Z = np.zeros((l, l))
    U = np.zeros((l, l))
    relax = 1.4
    O_old = O
    
#    rho = rho/l
    for it_o in range(max_iter):
        v, Q = eigh(rho*(Z - U) - beta/2*A)
        o = (v + np.sqrt(v**2 + 4*rho))/(2*rho)
        O = Q@np.diag(o)@Q.T
        
        O_hat = relax*O + (1 - relax)*Z
        Z = soft_threshold(O_hat + U, lam2/rho)
        
        U = U + (O_hat - Z)
    
    diff_O = norm(Z - O_old, 'fro')
    return O, Z, diff_O


def adaptive_neighbor(Ws, Xs, distXs, mu, gam, k, max_iter = 10):
    num_sample = Xs[0].shape[1]
    num_view = len(Xs)
    
    distX = np.zeros((num_sample, num_sample))
    for i in range(num_view):
        distX += distXs[i]
    distX = distX/num_view
    
    Fs = [Xs[i].T@Ws[i] for i in range(num_view)]
    distf = squareform(pdist(Fs[0], 'cosine')) + squareform(pdist(Fs[1], 'cosine'))
    distXf = mu*distX + gam/4*distf
    idx = np.argsort(distXf, 1)
    S = np.zeros((num_sample, num_sample))
    for i in range(num_sample):
        idxa0 = idx[i,1:k+2]
        di = distXf[i,idxa0]
        S[i,idxa0] = (di[k] - di + eps)/(k*di[k] - np.sum(di[0:k]) + k*eps)
    
    for it in range(max_iter):
        distX = np.zeros((num_sample, num_sample))
        wv = []
        for i in range(num_view):
            wv.append(0.5/np.sqrt(np.sum(distXs[i]*S)))
            distX += wv[i]*distXs[i]
#            print(wv[i]*distXs[i])
        
        distXf = mu*distX + gam/4*distf
        idx = np.argsort(distXf, 1)
        S = np.zeros((num_sample, num_sample))
        for i in range(num_sample):
            idxa0 = idx[i,1:k+2]
            di = distXf[i,idxa0]
            S[i,idxa0] = (di[k] - di + eps)/(k*di[k] - np.sum(di[0:k]) + k*eps)
    
    S_hat = (S + S.T)/2
    D = np.diag(np.sum(S_hat, 1))
    L = D - S_hat
    return S, L


def my_dist(X, dist_type):
    num_sample = X.shape[1]
    distX = squareform(pdist(X.T, dist_type))
    idx = np.argsort(distX, 1)
    distX_sorted = np.array([distX[i,idx[i,:]] for i in range(num_sample)])
    return distX, idx, distX_sorted


def agmsl(Xs, Y, param):
    max_iter = param['max_iter']
    tol = param['tol']
    gam = param['gam']
    beta = param['beta']
    lam1 = param['lam1']
    lam2 = param['lam2']
    mu = param['mu']
    k = param['k']
    rho = param['rho']

    num_feat = [Xs[i].shape[0] for i in range(len(Xs))]
    num_label, num_sample = Y.shape

    O = np.eye(num_label)
    Z = np.eye(num_label)
    W1 = inv(Xs[0]@Xs[0].T + 0.1*np.eye(num_feat[0]))@Xs[0]@Y.T
    W2 = inv(Xs[1]@Xs[1].T + 0.1*np.eye(num_feat[1]))@Xs[1]@Y.T
    # W3 = inv(Xs[2]@Xs[2].T + 0.1*np.eye(num_feat[2]))@Xs[2]@Y.T
    Ws = [W1, W2]

    distX1, idx1, distX_sorted1 = my_dist(Xs[0], 'cosine')
    distX2, idx2, distX_sorted2 = my_dist(Xs[1], 'sqeuclidean')
    # distX3, idx3, distX_sorted3 = my_dist(Xs[2], 'sqeuclidean')   
    distXs = [distX1, distX2]
    
    S, L = adaptive_neighbor(Ws, Xs, distXs, mu, gam, k)
    
    for it in range(max_iter):
        W1, diff_W1 = acc_proximal_gradient(W1, Xs[0], Y, L, Z, gam, beta, lam1, max_iter = 10)
        W2, diff_W2 = acc_proximal_gradient(W2, Xs[1], Y, L, Z, gam, beta, lam1, max_iter = 10)
        # W3, diff_W3 = acc_proximal_gradient(W3, Xs[2], Y, L, Z, gam, beta, lam1, max_iter = 10)
        
        Ws = [W1, W2]
        O, Z, diff_O = covsel_admm(Z, Ws, rho, beta, lam2, 10)
        
        S, L = adaptive_neighbor(Ws, Xs, distXs, mu, gam, k)
        
        obj_val = obj_fun(Ws, Xs, distXs, Y, L, S, O, param)
        print(it, diff_W1, diff_W2, diff_O, obj_val)
        
        if diff_W1 < tol:
            break
    
    return Ws, O, Z, S, L, it


#%%
drug2cui_tab = pd.read_csv('./drug2cui_tab_st.csv', header = 0, index_col = 0)
drug2fp = pd.read_csv('./drug_FCFP4_fingerprint.csv', header = 0, index_col = 0)
drug2path = pd.read_csv('./drug_stitch_gsva_kegg.csv', header = 0, index_col = 0)

drug2cui_tab_test = pd.read_csv('./drug2cui_tab_st_independent.csv', header = 0, index_col = 0)
drug2fp_test = pd.read_csv('./drug_FCFP4_fingerprint_independent.csv', header = 0, index_col = 0)
drug2path_test = pd.read_csv('./drug_stitch_gsva_kegg_independent.csv', header = 0, index_col = 0)

#%%
Ytrain = drug2cui_tab.T.to_numpy()
X1_train = drug2fp.T.to_numpy()
X2_train = drug2path.T.to_numpy()
# X3_all = drug2expr.T.to_numpy()
Xtrain = [X1_train, X2_train]
ntrain = Ytrain.shape[1]

Ytest = drug2cui_tab_test.T.to_numpy()
X1_test = drug2fp_test.T.to_numpy()
X2_test = drug2path_test.T.to_numpy()
# X3_all = drug2expr.T.to_numpy()
Xtest = [X1_test, X2_test]
ntest = Ytest.shape[1]

param = {}
param['max_iter'] = 200
param['tol'] = 1e-4
param['gam'] = 2
param['beta'] = 5
param['lam1'] = 0.2
param['lam2'] = 0.00005
param['mu'] = 1e-4
param['k'] = 9
param['rho'] = 2

#%%

Xtrain = [np.vstack((Xtrain[i], np.ones(ntrain))) for i in range(len(Xtrain))]
Xtest = [np.vstack((Xtest[i], np.ones(ntest))) for i in range(len(Xtest))]
   
Ws, O, Z, S, L, num_iter = agmsl(Xtrain, Ytrain, param)
score = Ws[0].T@Xtest[0] + Ws[1].T@Xtest[1]
auc = roc_auc_score(Ytest.T, score.T, average = 'samples')
aupr = average_precision_score(Ytest.T, score.T, average = 'samples')
co_error = coverage_error(Ytest.T, score.T)
lrap = label_ranking_average_precision_score(Ytest.T, score.T)
rloss = label_ranking_loss(Ytest.T, score.T)
print(auc, aupr, co_error, lrap, rloss)

