import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
import scipy
from datasets import fetch_data_generator

def instance(nu, U, y):
    n = y.shape[0]
    ploss = np.mean((y.reshape(1, -1) - U)**2, axis=1)

    def loss(x):
        return np.mean((y - U.T @ x)**2)

    def qfunction(x):
        return (1 - nu) * loss(x) + nu * x @ ploss

    def grad_q(x):
        return - 2 * (1 - nu) * U @ (y - U.T @ x) / n + nu * ploss

    return loss, qfunction, grad_q, ploss

def opt(K, qfunction, grad_q):
    res = scipy.optimize.minimize(qfunction, np.ones(K)/K, jac=grad_q, bounds=[(0, 1)]*K,
                                  constraints=scipy.optimize.LinearConstraint(np.ones((1, K)), lb=1, ub=1),
                                  tol=1e-12)
    return res.x

def qagg(F, y):
    scale = max(np.max(np.abs(F)), np.max(np.abs(y)))
    loss, qfunction, grad_q, ploss = instance(.5, F.T / scale, y / scale)
    return opt(F.shape[1], qfunction, grad_q)

def mse(y, yhat):
    return np.mean((y - yhat)**2)

def get_dgp(dgp):
    if dgp == 1:
        # dgp that favors the x-learner
        prop = lambda z: .05 * np.ones(z.shape)
        base = lambda z: .1 * (z<=.8) * (z>=.6)
        cate = lambda z: .5 * np.ones(z.shape)
    
    if dgp == 2:
        # dgp that favors the dr-learner
        prop = lambda z: .05 * np.ones(z.shape)
        base = lambda z: .1 * (z<=.8) * (z>=.6)
        cate = lambda z: (.5 + .1 * (z<=.4) * (z>=.2))

    if dgp == 3:
        # dgp that favors the r-learner
        prop = lambda z: .5 - .49 * (z<=.6) * (z>=.3)
        base = lambda z: .1 * (z<=.8) * (z>=.6)
        cate = lambda z: .5 * z**2

    if dgp == 4:
        # dgp that favors the covariate shift X-learner over X-learner
        prop = lambda z: .5 - .49 * (z<=.6) * (z>=.3)
        base = lambda z: .1 * (z<=.8) * (z>=.6)
        cate = lambda z: .5 * z**2 * (z<=.6) + .18
    
    if dgp == 5:
        # dgp that removes superiority of X-learner strategy
        prop = lambda z: .05 * np.ones(z.shape)
        base = lambda z: .1 * np.ones(z.shape)
        cate = lambda z: .5 * (z<=.8) * (z>=.5)
    
    if dgp == 6:
        # dgp that favors the opposite of the X-learner strategy
        prop = lambda z: .95 * np.ones(z.shape)
        base = lambda z: .1 * np.ones(z.shape)
        cate = lambda z: .5 * (z<=.8) * (z>=.5)

    return prop, base, cate

def gen_data(dgp, *, n=None, semi_synth=False, simple_synth=False,
             scale=1, true_f=None, max_depth=3, random_state=123):
    
    np.random.seed(random_state)
    if dgp in list(np.arange(1, 7)):
        prop, base, cate = get_dgp(dgp)

        def get_data(n):
            Z = np.random.uniform(0, 1, size=n)
            D = np.random.binomial(1, prop(Z))
            Y = cate(Z) * D + base(Z) + np.random.normal(0, .05, size=n)
            return Z.reshape(-1, 1), D, Y
        
        Z, D, Y = get_data(n)
        Zval, Dval, Yval = get_data(n)
        Ztest, Dtest, Ytest = get_data(10000)
        inds = np.argsort(Ztest.flatten())
        Ztest, Dtest, Ytest = Ztest[inds], Dtest[inds], Ytest[inds]
        true_cate = lambda z: cate(z).flatten()
    else:
        get_data, abtest, true_f, true_cate = fetch_data_generator(data=dgp,
                                                              semi_synth=semi_synth,
                                                              simple_synth=simple_synth,
                                                              scale=scale,
                                                              true_f=true_f,
                                                              max_depth=max_depth,
                                                              random_state=random_state)
        Z, D, Y, _ = get_data()
        Z = np.array(Z.values)
        Z, Zval, D, Dval, Y, Yval = train_test_split(Z, D, Y, test_size=.6)
        Zval, Ztest, Dval, Dtest, Yval, Ytest = train_test_split(Zval, Dval, Yval, test_size=.5)

    return (Z, D, Y), (Zval, Dval, Yval), (Ztest, Dtest, Ytest), true_cate


def experiment(dgp, *, n=None, semi_synth=False, simple_synth=False,
               scale=1, true_f=None, max_depth=3, random_state=123):

    np.random.seed(random_state)
    dtrain, dval, dtest, cate = gen_data(dgp, n=n, semi_synth=semi_synth, simple_synth=simple_synth,
                                         scale=scale, true_f=true_f, max_depth=max_depth,
                                         random_state=random_state)
    Z, D, Y = dtrain
    Zval, Dval, Yval = dval
    Ztest, _, _ = dtest

    param_grid = {
        'max_depth': [2, 5, 7],
        'min_samples_leaf': [20, 50]
    }
    reg = lambda: GridSearchCV(RandomForestRegressor(random_state=123), param_grid)
    clf = lambda: GridSearchCV(RandomForestClassifier(random_state=123), param_grid)

    #############################################
    #### Train on training set all meta-learners
    #############################################
    g0 = reg().fit(Z[D==0], Y[D==0])
    g1 = reg().fit(Z[D==1], Y[D==1])

    # propensity
    mu = clf().fit(Z, D)

    # X-learner
    m = mu.predict_proba(Z)[:, 1]
    tau0 = reg().fit(Z[D==0], g1.predict(Z[D==0]) - Y[D==0])
    tau1 = reg().fit(Z[D==1], Y[D==1] - g0.predict(Z[D==1]))

    # S-Learner
    g = reg().fit(np.hstack([D.reshape(-1, 1), Z]), Y)
    
    # IPS-Learner
    m = mu.predict_proba(Z)[:, 1]
    cov = np.clip(m * (1 - m), 1e-12, np.inf)
    tauIPS = reg().fit(Z, Y * (D - m)/cov)
    
    # DR-Learner
    m = mu.predict_proba(Z)[:, 1]
    cov = np.clip(m * (1 - m), 1e-12, np.inf)
    g1preds = g1.predict(Z)
    g0preds = g0.predict(Z)
    gpreds = g1preds * D + g0preds * (1 - D)
    Ydr = (Y - gpreds) * (D - m)/cov + g1preds - g0preds
    tauDR = reg().fit(Z, Ydr)

    # R-Learner
    h = reg().fit(Z, Y)
    Yres = Y - h.predict(Z)
    Dres = D - mu.predict_proba(Z)[:, 1]
    DresClip = np.clip(Dres, 1e-12, np.inf) * (Dres >= 0) + np.clip(Dres, -np.inf, -1e-12) * (Dres<0)
    tauR = reg().fit(Z, Yres/DresClip, sample_weight=Dres**2)

    # DRX-Learner
    m = mu.predict_proba(Z)[:, 1]
    g0preds = g0.predict(Z)
    g1preds = g1.predict(Z)
    g0preds = g0preds * (1 - m) + (g1.predict(Z) - tau0.predict(Z)) * m
    g1preds = g1preds * m + (g0.predict(Z) + tau1.predict(Z)) * (1 - m)
    gpreds = g1preds * D + g0preds * (1 - D)
    cov = np.clip(m * (1 - m), 1e-12, np.inf)
    Ydr = (Y - gpreds) * (D - m)/cov + g1preds - g0preds
    tauDRX = reg().fit(Z, Ydr)

    ####################################################
    ### Evaluate on validation and construct ensemble
    ####################################################
    weightsQ, weightsBest = {}, {}
    for name, train_on_val, cfit_on_val in [('train', False, False),
                                            ('val', True, False), 
                                            ('cfit', True, True)]:
        if train_on_val:
            g0best = clone(g0.best_estimator_)
            g1best = clone(g1.best_estimator_)
            mubest = clone(mu.best_estimator_)
            if cfit_on_val:
                m = np.zeros(Zval.shape[0])
                g1preds = np.zeros(Zval.shape[0])
                g0preds = np.zeros(Zval.shape[0])
                for train, test in KFold(n_splits=5).split(Zval, Yval):
                    g0val = g0best.fit(Zval[train][Dval[train]==0], Yval[train][Dval[train]==0])
                    g1val = g1best.fit(Zval[train][Dval[train]==1], Yval[train][Dval[train]==1])
                    muval = mubest.fit(Zval[train], Dval[train])
                    m[test] = muval.predict_proba(Zval[test])[:, 1]
                    g1preds[test] = g1val.predict(Zval[test])
                    g0preds[test] = g0val.predict(Zval[test])
            else:
                g0val = g0best.fit(Zval[Dval==0], Yval[Dval==0])
                g1val = g1best.fit(Zval[Dval==1], Yval[Dval==1])
                muval = mubest.fit(Zval, Dval)
                m = muval.predict_proba(Zval)[:, 1]
                g1preds = g1val.predict(Zval)
                g0preds = g0val.predict(Zval)
        else:
            m = mu.predict_proba(Zval)[:, 1]
            g1preds = g1.predict(Zval)
            g0preds = g0.predict(Zval)

        # DR-Score target labels
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        gpreds = g1preds * Dval + g0preds * (1 - Dval)
        Ydrval = (Yval - gpreds) * (Dval - m) / cov + g1preds - g0preds

        t0 = tau0.predict(Zval)
        t1 = tau1.predict(Zval)
        tT = g1.predict(Zval) - g0.predict(Zval)
        tS = g.predict(np.hstack([np.ones((Zval.shape[0], 1)), Zval]))
        tS -= g.predict(np.hstack([np.zeros((Zval.shape[0], 1)), Zval]))
        tIPS = tauIPS.predict(Zval)
        tDR = tauDR.predict(Zval)
        tR = tauR.predict(Zval)
        m = mu.predict_proba(Zval)[:, 1]
        tX = t1 * (1 - m) + t0 * m
        tDRX = tauDRX.predict(Zval)

        F = np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX), -1)
        weightsQ[name] = qagg(F, Ydrval)
        weightsBest[name] = np.zeros(weightsQ[name].shape)
        weightsBest[name][np.argmin(np.mean((Ydrval.reshape(-1, 1) - F)**2, axis=0))] = 1.0
    
    ####################################################
    # Final evaluation on test set
    ####################################################
    t0 = tau0.predict(Ztest)
    t1 = tau1.predict(Ztest)
    tT = g1.predict(Ztest) - g0.predict(Ztest)
    tS = g.predict(np.hstack([np.ones((Ztest.shape[0], 1)), Ztest]))
    tS -= g.predict(np.hstack([np.zeros((Ztest.shape[0], 1)), Ztest]))
    tIPS = tauIPS.predict(Ztest)
    tDR = tauDR.predict(Ztest)
    tR = tauR.predict(Ztest)
    m = mu.predict_proba(Ztest)[:, 1]
    tX = t1 * (1 - m) + t0 * m
    tDRX = tauDRX.predict(Ztest)

    mses = {'T': mse(cate(Ztest), tT),
            'S': mse(cate(Ztest), tS),
            'IPS': mse(cate(Ztest), tIPS),
            'DR': mse(cate(Ztest), tDR),
            'R': mse(cate(Ztest), tR),
            'X': mse(cate(Ztest), tX),
            'DRX': mse(cate(Ztest), tDRX)}
    cates = {'T': tT,
            'S': tS,
            'IPS': tIPS,
            'DR': tDR,
            'R': tR,
            'X': tX,
            'DRX': tDRX,
            'True': cate(Ztest),
            'Ztest': Ztest}

    F = np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX), -1)
    tUniform = F @ np.ones(F.shape[1]) / F.shape[1]
    mses['Uni'] = mse(cate(Ztest), tUniform)
    cates['Uni'] = tUniform
    for name in ['train', 'val', 'cfit']:
        tQ = F @ weightsQ[name]
        tBest = F @ weightsBest[name]
        mses[f'Q{name}'] = mse(cate(Ztest), tQ)
        mses[f'Best{name}'] = mse(cate(Ztest), tBest)
        cates[f'Q{name}'] = tQ
        cates[f'Best{name}'] = tBest

    return mses, cates