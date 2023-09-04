from sys import meta_path
import numpy as np
from numpy.lib.ufunclike import isposinf
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
    res = scipy.optimize.minimize(qfunction, np.ones(K) / K, jac=grad_q, bounds=[(0, 1)] * K,
                                  constraints=scipy.optimize.LinearConstraint(
                                      np.ones((1, K)), lb=1, ub=1),
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
        def prop(z): return .05 * np.ones(z.shape)

        def base(z): return .1 * (z <= .8) * (z >= .6)

        def cate(z): return .5 * np.ones(z.shape)

    if dgp == 2:
        # dgp that favors the dr-learner
        def prop(z): return .05 * np.ones(z.shape)

        def base(z): return .1 * (z <= .8) * (z >= .6)

        def cate(z): return (.5 + .1 * (z <= .4) * (z >= .2))

    if dgp == 3:
        # dgp that favors the r-learner
        def prop(z): return .5 - .49 * (z <= .6) * (z >= .3)

        def base(z): return .1 * (z <= .8) * (z >= .6)

        def cate(z): return .5 * z**2

    if dgp == 4:
        # dgp that favors the covariate shift X-learner over X-learner
        def prop(z): return .5 - .49 * (z <= .6) * (z >= .3)

        def base(z): return .1 * (z <= .8) * (z >= .6)

        def cate(z): return .5 * z**2 * (z <= .6) + .18

    if dgp == 5:
        # dgp that removes superiority of X-learner strategy
        def prop(z): return .05 * np.ones(z.shape)

        def base(z): return .1 * np.ones(z.shape)

        def cate(z): return .5 * (z <= .8) * (z >= .5)

    if dgp == 6:
        # dgp that favors the opposite of the X-learner strategy
        def prop(z): return .95 * np.ones(z.shape)

        def base(z): return .1 * np.ones(z.shape)

        def cate(z): return .5 * (z <= .8) * (z >= .5)

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

        def true_cate(z): return cate(z).flatten()
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
        Zval, Ztest, Dval, Dtest, Yval, Ytest = train_test_split(
            Zval, Dval, Yval, test_size=.5)

    return (Z, D, Y), (Zval, Dval, Yval), (Ztest, Dtest, Ytest), true_cate


class MetaLearners:

    def __init__(self, reg, clf):
        self.reg = reg
        self.clf = clf

    def fit(self, Z, D, Y):
        reg = self.reg
        clf = self.clf

        self.g0 = reg().fit(Z[D == 0], Y[D == 0])
        self.g1 = reg().fit(Z[D == 1], Y[D == 1])

        # propensity
        self.mu = clf().fit(Z, D)

        # X-learner
        m = self.mu.predict_proba(Z)[:, 1]
        self.tau0 = reg().fit(
            Z[D == 0], self.g1.predict(Z[D == 0]) - Y[D == 0])
        self.tau1 = reg().fit(Z[D == 1], Y[D == 1] -
                              self.g0.predict(Z[D == 1]))

        # S-Learner
        self.g = reg().fit(np.hstack([D.reshape(-1, 1), Z]), Y)

        # IPS-Learner
        m = self.mu.predict_proba(Z)[:, 1]
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        self.tauIPS = reg().fit(Z, Y * (D - m) / cov)

        # DR-Learner
        m = self.mu.predict_proba(Z)[:, 1]
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        g1preds = self.g1.predict(Z)
        g0preds = self.g0.predict(Z)
        gpreds = g1preds * D + g0preds * (1 - D)
        Ydr = (Y - gpreds) * (D - m) / cov + g1preds - g0preds
        self.tauDR = reg().fit(Z, Ydr)

        # R-Learner
        self.h = reg().fit(Z, Y)
        Yres = Y - self.h.predict(Z)
        Dres = D - self.mu.predict_proba(Z)[:, 1]
        DresClip = np.clip(Dres, 1e-12, np.inf) * (Dres >= 0) + \
            np.clip(Dres, -np.inf, -1e-12) * (Dres < 0)
        self.tauR = reg().fit(Z, Yres / DresClip, sample_weight=Dres**2)

        # DRX-Learner
        m = self.mu.predict_proba(Z)[:, 1]
        g0preds = self.g0.predict(Z)
        g1preds = self.g1.predict(Z)
        g0preds = g0preds * (1 - m) + \
            (self.g1.predict(Z) - self.tau0.predict(Z)) * m
        g1preds = g1preds * m + \
            (self.g0.predict(Z) + self.tau1.predict(Z)) * (1 - m)
        gpreds = g1preds * D + g0preds * (1 - D)
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        Ydr = (Y - gpreds) * (D - m) / cov + g1preds - g0preds
        self.tauDRX = reg().fit(Z, Ydr)

        return self

    def predict(self, Z):
        ''' returns [T, S, IPS, DR, R, X, DRX] CATE predictions
        '''
        t0 = self.tau0.predict(Z)
        t1 = self.tau1.predict(Z)
        tT = self.g1.predict(Z) - self.g0.predict(Z)
        tS = self.g.predict(np.hstack([np.ones((Z.shape[0], 1)), Z]))
        tS -= self.g.predict(np.hstack([np.zeros((Z.shape[0], 1)), Z]))
        tIPS = self.tauIPS.predict(Z)
        tDR = self.tauDR.predict(Z)
        tR = self.tauR.predict(Z)
        m = self.mu.predict_proba(Z)[:, 1]
        tX = t1 * (1 - m) + t0 * m
        tDRX = self.tauDRX.predict(Z)

        return np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX), -1)


class Ensemble:

    def __init__(self, *, meta, train_on_val=True, cfit_on_val=False, three_way=False):
        self.meta = meta
        self.train_on_val = train_on_val
        self.cfit_on_val = cfit_on_val
        self.three_way = three_way

    def fit(self, Zval, Dval, Yval):
        meta = self.meta
        if self.train_on_val:
            g0best = clone(meta.g0.best_estimator_)
            g1best = clone(meta.g1.best_estimator_)
            mubest = clone(meta.mu.best_estimator_)
            if self.cfit_on_val:
                m = np.zeros(Zval.shape[0])
                g1preds = np.zeros(Zval.shape[0])
                g0preds = np.zeros(Zval.shape[0])
                for train, test in KFold(n_splits=5).split(Zval, Yval):
                    nt = len(train)
                    regfold = train[nt // 2:] if self.three_way == 2 else train
                    rrfold = train[:nt // 2] if self.three_way == 2 else train
                    g0val = g0best.fit(
                        Zval[regfold][Dval[regfold] == 0], Yval[regfold][Dval[regfold] == 0])
                    g1val = g1best.fit(
                        Zval[regfold][Dval[regfold] == 1], Yval[regfold][Dval[regfold] == 1])
                    muval = mubest.fit(Zval[rrfold], Dval[rrfold])
                    m[test] = muval.predict_proba(Zval[test])[:, 1]
                    g1preds[test] = g1val.predict(Zval[test])
                    g0preds[test] = g0val.predict(Zval[test])
            else:
                g0val = g0best.fit(Zval[Dval == 0], Yval[Dval == 0])
                g1val = g1best.fit(Zval[Dval == 1], Yval[Dval == 1])
                muval = mubest.fit(Zval, Dval)
                m = muval.predict_proba(Zval)[:, 1]
                g1preds = g1val.predict(Zval)
                g0preds = g0val.predict(Zval)
        else:
            m = meta.mu.predict_proba(Zval)[:, 1]
            g1preds = meta.g1.predict(Zval)
            g0preds = meta.g0.predict(Zval)

        # DR-Score target labels
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        gpreds = g1preds * Dval + g0preds * (1 - Dval)
        Ydrval = (Yval - gpreds) * (Dval - m) / cov + g1preds - g0preds

        F = meta.predict(Zval)
        self.weightsQ_ = qagg(F, Ydrval)
        self.weightsBest_ = np.zeros(self.weightsQ_.shape)
        self.weightsBest_[np.argmin(
            np.mean((Ydrval.reshape(-1, 1) - F)**2, axis=0))] = 1.0

        return self

    def predictQ(self, Ztest):
        F = self.meta.predict(Ztest)
        return F @ self.weightsQ_

    def predictBest(self, Ztest):
        F = self.meta.predict(Ztest)
        return F @ self.weightsBest_


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

    def reg(): return GridSearchCV(RandomForestRegressor(random_state=123), param_grid)

    def clf(): return GridSearchCV(RandomForestClassifier(random_state=123), param_grid)

    #############################################
    # Train on training set all meta-learners
    #############################################
    print('Fitting meta learners on dtrain')
    meta = MetaLearners(reg, clf).fit(Z, D, Y)

    ####################################################
    # Evaluate on validation and construct ensemble
    ####################################################
    ensemble = {}
    for name, train_on_val, cfit_on_val, three_way in [('train', False, False, False),
                                                       ('val', True, False, False),
                                                       ('cfit', True, True, False),
                                                       ('3way', True, True, True)]:
        print(f'Fitting ensemble {name} on dval')
        ensemble[name] = Ensemble(
            meta=meta, train_on_val=train_on_val, cfit_on_val=cfit_on_val, three_way=three_way)
        ensemble[name].fit(Zval, Dval, Yval)

    ####################################################
    # Meta learners on all dtrain and dval
    ####################################################
    print('Fitting meta learners on dtrain union dval')
    meta_all = MetaLearners(reg, clf).fit(
        np.vstack([Z, Zval]), np.concatenate((D, Dval)), np.concatenate((Y, Yval)))

    ####################################################
    # Final evaluation on test set
    ####################################################
    print('Evaluating all models on dtest')
    F = meta.predict(Ztest)

    mses = {'T': mse(cate(Ztest), F[:, 0]),
            'S': mse(cate(Ztest), F[:, 1]),
            'IPS': mse(cate(Ztest), F[:, 2]),
            'DR': mse(cate(Ztest), F[:, 3]),
            'R': mse(cate(Ztest), F[:, 4]),
            'X': mse(cate(Ztest), F[:, 5]),
            'DRX': mse(cate(Ztest), F[:, 6])}
    cates = {'T': F[:, 0],
             'S': F[:, 1],
             'IPS': F[:, 2],
             'DR': F[:, 3],
             'R': F[:, 4],
             'X': F[:, 5],
             'DRX': F[:, 6],
             'True': cate(Ztest),
             'Ztest': Ztest}

    tUniform = F @ np.ones(F.shape[1]) / F.shape[1]
    mses['Uni'] = mse(cate(Ztest), tUniform)
    cates['Uni'] = tUniform
    for name in ['train', 'val', 'cfit', '3way']:
        tQ = ensemble[name].predictQ(Ztest)
        tBest = ensemble[name].predictBest(Ztest)
        mses[f'Q{name}'] = mse(cate(Ztest), tQ)
        mses[f'Best{name}'] = mse(cate(Ztest), tBest)
        cates[f'Q{name}'] = tQ
        cates[f'Best{name}'] = tBest

    F = meta_all.predict(Ztest)
    mses = {**mses,
            'Tall': mse(cate(Ztest), F[:, 0]),
            'Sall': mse(cate(Ztest), F[:, 1]),
            'IPSall': mse(cate(Ztest), F[:, 2]),
            'DRall': mse(cate(Ztest), F[:, 3]),
            'Rall': mse(cate(Ztest), F[:, 4]),
            'Xall': mse(cate(Ztest), F[:, 5]),
            'DRXall': mse(cate(Ztest), F[:, 6])}
    cates = {**cates,
             'Tall': F[:, 0],
             'Sall': F[:, 1],
             'IPSall': F[:, 2],
             'DRall': F[:, 3],
             'Rall': F[:, 4],
             'Xall': F[:, 5],
             'DRXall': F[:, 6]}

    return mses, cates
