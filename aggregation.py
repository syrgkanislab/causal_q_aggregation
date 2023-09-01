import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy

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

def experiment(n, dgp, random_state):
    np.random.seed(random_state)
    Z = np.random.uniform(0, 1, size=n)
    prop, base, cate = get_dgp(dgp)
    D = np.random.binomial(1, prop(Z))
    Y = cate(Z) * D + base(Z) + np.random.normal(0, .05, size=n)

    param_grid = {
        'max_depth': [2],
        'min_samples_leaf': [50]
    }
    reg = lambda: GridSearchCV(RandomForestRegressor(random_state=123), param_grid)
    clf = lambda: GridSearchCV(RandomForestClassifier(random_state=123), param_grid)

    Z = Z.reshape(-1, 1)
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



    Zval = np.random.uniform(0, 1, size=n)
    Dval = np.random.binomial(1, prop(Zval))
    Yval = cate(Zval) * Dval + base(Zval) + np.random.normal(0, .05, size=n)
    Zval = Zval.reshape(-1, 1)
    g0val = reg().fit(Zval[Dval==0], Yval[Dval==0])
    g1val = reg().fit(Zval[Dval==1], Yval[Dval==1])
    muval = clf().fit(Zval, Dval)
    # DR-Learner
    m = muval.predict_proba(Zval)[:, 1]
    cov = np.clip(m * (1 - m), 1e-12, np.inf)
    g1preds = g1val.predict(Zval)
    g0preds = g0val.predict(Zval)
    gpreds = g1preds * Dval + g0preds * (1 - Dval)
    Ydrval = (Yval - gpreds) * (Dval - m)/cov + g1preds - g0preds

    t0 = tau0.predict(Zval)
    t1 = tau1.predict(Zval)
    tT = g1.predict(Zval) - g0.predict(Zval)
    tS = g.predict(np.stack((np.ones(n), Zval.flatten()), -1))
    tS -= g.predict(np.stack((np.zeros(n), Zval.flatten()), -1))
    tIPS = tauIPS.predict(Zval)
    tDR = tauDR.predict(Zval)
    tR = tauR.predict(Zval)
    m = mu.predict_proba(Zval)[:, 1]
    tX = t1 * (1 - m) + t0 * m
    tDRX = tauDRX.predict(Zval)

    F = np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX), -1)
    weightsQ = qagg(F, Ydrval)
    weightsBest = np.zeros(weightsQ.shape)
    weightsBest[np.argmin(np.mean((Ydrval.reshape(-1, 1) - F)**2, axis=0))] = 1.0
    
    # Evaluation on Test
    ntest = 10000
    Ztest = np.random.uniform(0, 1, size=ntest)
    Ztest = np.sort(Ztest)
    t0 = tau0.predict(Ztest.reshape(-1, 1))
    t1 = tau1.predict(Ztest.reshape(-1, 1))
    tT = g1.predict(Ztest.reshape(-1, 1)) - g0.predict(Ztest.reshape(-1, 1))
    tS = g.predict(np.stack((np.ones(ntest), Ztest), -1)) - g.predict(np.stack((np.zeros(ntest), Ztest), -1))
    tIPS = tauIPS.predict(Ztest.reshape(-1, 1))
    tDR = tauDR.predict(Ztest.reshape(-1, 1))
    tR = tauR.predict(Ztest.reshape(-1, 1))
    m = mu.predict_proba(Ztest.reshape(-1, 1))[:, 1]
    tX = t1 * (1 - m) + t0 * m
    tDRX = tauDRX.predict(Ztest.reshape(-1, 1))
    F = np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX), -1)
    tQ = F @ weightsQ
    tBest = F @ weightsBest

    mses = {'T': mse(cate(Ztest), tT),
               'S': mse(cate(Ztest), tS),
               'IPS': mse(cate(Ztest), tIPS),
               'DR': mse(cate(Ztest), tDR),
               'R': mse(cate(Ztest), tR),
               'X': mse(cate(Ztest), tX),
               'DRX': mse(cate(Ztest), tDRX),
               'Q': mse(cate(Ztest), tQ),
               'B': mse(cate(Ztest), tBest)}

    cates = {'T': tT,
            'S': tS,
            'IPS': tIPS,
            'DR': tDR,
            'R': tR,
            'X': tX,
            'DRX': tDRX,
            'Q': tQ,
            'B': tBest,
            'True': cate(Ztest),
            'Ztest': Ztest}

    return mses, cates