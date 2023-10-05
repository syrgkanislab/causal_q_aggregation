from sys import meta_path
import numpy as np
from numpy.lib.ufunclike import isposinf
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.base import clone
import scipy
from datasets import fetch_data_generator
from econml.grf import CausalForest, RegressionForest
from myxgb import xgb_reg, xgb_clf, xgb_wreg


def instance(nu, U, y, prior=None, beta=None):
    n = y.shape[0]
    ploss = np.mean((y.reshape(1, -1) - U)**2, axis=1)

    if prior is not None:
        loginvprior = np.log(1/prior)

    def loss(x):
        return np.mean((y - U.T @ x)**2)
    
    if prior is not None:    
        def qfunction(x):
            return (1 - nu) * loss(x) + nu * x @ ploss + beta * x @ loginvprior / n
    
        def grad_q(x):
            return - 2 * (1 - nu) * U @ (y - U.T @ x) / n + nu * ploss + beta * loginvprior / n

    else:
        def qfunction(x):
            return (1 - nu) * loss(x) + nu * x @ ploss

        def grad_q(x):
            return - 2 * (1 - nu) * U @ (y - U.T @ x) / n + nu * ploss

    return loss, qfunction, grad_q, ploss


def opt(K, qfunction, grad_q):
    res = scipy.optimize.minimize(qfunction, np.ones(K) / K, jac=grad_q, bounds=[(0, 1)] * K,
                                  constraints=scipy.optimize.LinearConstraint(
                                      np.ones((1, K)), lb=1, ub=1),
                                  tol=1e-14)
    return res.x


def qagg(F, y, *, nu=.5, prior=None, beta=None):
    scale = max(np.max(np.abs(F)), np.max(np.abs(y)))
    loss, qfunction, grad_q, ploss = instance(nu, F.T / scale, y / scale, prior=prior, beta=beta)
    return opt(F.shape[1], qfunction, grad_q)


def mse(y, yhat):
    return np.mean((y.flatten() - yhat.flatten())**2)


def get_dgp(dgp):
    if dgp == 1:
        # dgp that favors the x-learner
        def prop(z): return .05 * np.ones(z.shape)

        def base(z): return .1 * (z <= .8) * (z >= .6)

        def cate(z): return .5 * np.ones(z.shape)

    if dgp == 2:
        # dgp that slightly favors the dr-learner over x-learner
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
            Y = cate(Z) * D + base(Z) + np.random.normal(0, scale, size=n)
            return Z.reshape(-1, 1), D, Y

        Z, D, Y = get_data(n)
        Zval, Dval, Yval = get_data(n//2)
        Ztest, Dtest, Ytest = get_data(10000)
        inds = np.argsort(Ztest.flatten())
        Ztest, Dtest, Ytest = Ztest[inds], Dtest[inds], Ytest[inds]

        def true_cate(z): return cate(z).flatten()
        def true_g0(z): return base(z).flatten()
        def true_g1(z): return base(z).flatten() + cate(z).flatten()
        def true_mu(z): return prop(z).flatten()
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
        true_g0, true_g1, true_mu = None, None, None

    return (Z, D, Y), (Zval, Dval, Yval), (Ztest, Dtest, Ytest), true_cate, true_g0, true_g1, true_mu


class MetaLearners:

    def __init__(self, reg, clf, wreg):
        self.reg = reg
        self.clf = clf
        self.wreg = wreg

    def fit(self, Z, D, Y):
        reg = self.reg
        clf = self.clf
        wreg = self.wreg

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
        self.tauR = wreg().fit(Z, Yres / DresClip, sample_weight=Dres**2)

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

        # DAX-Learner
        # correcting for covariate shift in CATE model estimation in X-Learner
        m = self.mu.predict_proba(Z)[:, 1]
        g0da = wreg().fit(Z[D==0], Y[D==0], sample_weight=(1 - m[D==0]))
        g1da = wreg().fit(Z[D==1], Y[D==1], sample_weight=m[D==1])
        self.tau0da = wreg().fit(Z[D==0], g1da.predict(Z[D==0]) - Y[D==0], sample_weight=m[D==0]**2 / (1 - m[D==0]))
        self.tau1da = wreg().fit(Z[D==1], Y[D==1] - g0da.predict(Z[D==1]), sample_weight=(1 - m[D==1])**2 / m[D==1])

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
        tDR = self.tauDR.predict(Z).flatten()
        tR = self.tauR.predict(Z).flatten()
        m = self.mu.predict_proba(Z)[:, 1]
        tX = t1 * (1 - m) + t0 * m
        tDRX = self.tauDRX.predict(Z)
        t0da = self.tau0da.predict(Z)
        t1da = self.tau1da.predict(Z)
        tXda = t1da * (1 - m) + t0da * m

        return np.stack((tT, tS, tIPS, tDR, tR, tX, tDRX, tXda), -1)
    
    def get_prior(self, prior_dict):
        prior = np.zeros(7)
        for it, name in enumerate(['T', 'S', 'IPS', 'DR', 'R', 'X', 'DRX', 'DAX']):
            prior[it] = prior_dict[name]
        return prior


class DRLearner:
    def __init__(self, reg, clf, cate_models):
        self.reg = reg
        self.clf = clf
        self.cate_models = cate_models

    def fit(self, Z, D, Y):
        reg = self.reg
        clf = self.clf

        self.g0 = reg().fit(Z[D == 0], Y[D == 0])
        self.g1 = reg().fit(Z[D == 1], Y[D == 1])

        # propensity
        self.mu = clf().fit(Z, D)

        # DR-Learner
        m = self.mu.predict_proba(Z)[:, 1]
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        g1preds = self.g1.predict(Z)
        g0preds = self.g0.predict(Z)
        gpreds = g1preds * D + g0preds * (1 - D)
        Ydr = (Y - gpreds) * (D - m) / cov + g1preds - g0preds
        self.models_ = [(name, clone(model).fit(Z, Ydr))
                        for name, model in self.cate_models]

        return self

    def predict(self, Z):
        ''' returns CATE predictions
        '''
        return np.stack([model.predict(Z) for _, model in self.models_], -1)

    def cate_names(self,):
        return [name for name, _ in self.models_]


class Ensemble:

    def __init__(self, *, meta, nuisance_mode,
                 model_select=True, g0=None, g1=None, mu=None,
                 prior=None, beta=None, nu = None):
        self.meta = meta
        self.nuisance_mode = nuisance_mode
        self.model_select = model_select
        self.g0 = g0
        self.g1 = g1
        self.mu = mu
        self.prior = prior
        self.beta = beta
        self.nu = nu

    def fit(self, Zval, Dval, Yval):
        meta = self.meta
        if self.nuisance_mode in ['train_on_val', 'split_on_val', 'cfit_on_val', '3way']:

            if self.model_select:
                g0best = clone(clone(meta.g0).fit(
                    Zval[Dval == 0], Yval[Dval == 0]).best_estimator_)
                g1best = clone(clone(meta.g1).fit(
                    Zval[Dval == 1], Yval[Dval == 1]).best_estimator_)
                mubest = clone(clone(meta.mu).fit(Zval, Dval).best_estimator_)
            else:
                g0best = clone(meta.g0)
                g1best = clone(meta.g1)
                mubest = clone(meta.mu)

            if self.nuisance_mode == 'split_on_val':
                Ztrain, Zval, Dtrain, Dval, Ytrain, Yval = train_test_split(
                    Zval, Dval, Yval, test_size=.3, shuffle=True, stratify=Dval)
                g0val = g0best.fit(Ztrain[Dtrain == 0], Ytrain[Dtrain == 0])
                g1val = g1best.fit(Ztrain[Dtrain == 1], Ytrain[Dtrain == 1])
                muval = mubest.fit(Ztrain, Dtrain)
                m = muval.predict_proba(Zval)[:, 1]
                g1preds = g1val.predict(Zval)
                g0preds = g0val.predict(Zval)

            if self.nuisance_mode in ['cfit_on_val', '3way']:
                m = np.zeros(Zval.shape[0])
                g1preds = np.zeros(Zval.shape[0])
                g0preds = np.zeros(Zval.shape[0])
                for train, test in StratifiedKFold(n_splits=5, shuffle=True).split(Zval, Dval):

                    if self.nuisance_mode == '3way':
                        regfold, rrfold = train_test_split(
                            train, test_size=.5, shuffle=True, stratify=Dval[train])
                    else:
                        regfold, rrfold = train, train

                    g0val = clone(g0best).fit(
                        Zval[regfold][Dval[regfold] == 0], Yval[regfold][Dval[regfold] == 0])
                    g1val = clone(g1best).fit(
                        Zval[regfold][Dval[regfold] == 1], Yval[regfold][Dval[regfold] == 1])
                    muval = clone(mubest).fit(Zval[rrfold], Dval[rrfold])
                    m[test] = muval.predict_proba(Zval[test])[:, 1]
                    g1preds[test] = g1val.predict(Zval[test])
                    g0preds[test] = g0val.predict(Zval[test])

            if self.nuisance_mode == 'train_on_val':
                g0val = g0best.fit(Zval[Dval == 0], Yval[Dval == 0])
                g1val = g1best.fit(Zval[Dval == 1], Yval[Dval == 1])
                muval = mubest.fit(Zval, Dval)
                m = muval.predict_proba(Zval)[:, 1]
                g1preds = g1val.predict(Zval)
                g0preds = g0val.predict(Zval)

        if self.nuisance_mode == 'from_train':
            m = meta.mu.predict_proba(Zval)[:, 1]
            g1preds = meta.g1.predict(Zval)
            g0preds = meta.g0.predict(Zval)

        if self.nuisance_mode == 'oracle':
            m = self.mu(Zval).flatten()
            g1preds = self.g1(Zval).flatten()
            g0preds = self.g0(Zval).flatten()

        # DR-Score target labels
        cov = np.clip(m * (1 - m), 1e-12, np.inf)
        gpreds = g1preds * Dval + g0preds * (1 - Dval)
        Ydrval = (Yval - gpreds) * (Dval - m) / cov + g1preds - g0preds

        F = meta.predict(Zval)
        self.weightsQ_ = qagg(F, Ydrval, prior=self.prior, beta=self.beta, nu=self.nu)
        self.weightsBest_ = np.zeros(self.weightsQ_.shape)
        self.weightsBest_[np.argmin(
            np.mean((Ydrval.reshape(-1, 1) - F)**2, axis=0))] = 1.0
        self.weightsConvex_ = qagg(F, Ydrval, prior=self.prior, beta=self.beta, nu=0)

        self.Zval = Zval.copy()
        self.g1preds = g1preds
        self.g0preds = g0preds
        self.mupreds = m

        return self

    def predictQ(self, Ztest):
        F = self.meta.predict(Ztest)
        return F @ self.weightsQ_

    def predictBest(self, Ztest):
        F = self.meta.predict(Ztest)
        return F @ self.weightsBest_

    def predictConvex(self, Ztest):
        F = self.meta.predict(Ztest)
        return F @ self.weightsConvex_


def experiment_dr(dgp, *, n=None, semi_synth=False, simple_synth=False,
                  scale=1, true_f=None, max_depth=3, random_state=123,
                  reg=xgb_reg,
                  clf=xgb_clf,
                  ensemble_methods=['train', 'val', 'split', 'cfit', '3way'],
                  model_select=False):
    print(random_state)
    np.random.seed(random_state)
    dtrain, dval, dtest, cate, g0, g1, mu = gen_data(dgp, n=n, semi_synth=semi_synth, simple_synth=simple_synth,
                                                     scale=scale, true_f=true_f, max_depth=max_depth,
                                                     random_state=random_state)
    Z, D, Y = dtrain
    Zval, Dval, Yval = dval
    Ztest, _, _ = dtest

    #############################################
    # Train on training set all meta-learners
    #############################################
    print('Fitting meta learners on dtrain')
    cate_models = [('lr', LinearRegression()),
                   ('ls1', Lasso(alpha=.001)),
                   ('ls2', Lasso(alpha=.01)),
                   ('ls3', Lasso(alpha=.1)),
                   ('ls4', Lasso(alpha=1)),
                   ('ls5', Lasso(alpha=10)),
                   ('enet1', ElasticNet(alpha=.001)),
                   ('enet2', ElasticNet(alpha=.01)),
                   ('enet3', ElasticNet(alpha=.1)),
                   ('enet4', ElasticNet(alpha=1)),
                   ('enet5', ElasticNet(alpha=10)),
                   ('poly1', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.001))])),
                   ('poly2', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.01))])),
                   ('poly3', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.1))])),
                   ('poly4', Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                       ('enet', ElasticNet(alpha=1))])),
                   ('poly5', Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.001))])),
                   ('poly6', Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.01))])),
                   ('poly7', Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                                       ('enet', ElasticNet(alpha=.1))])),
                   ('poly8', Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                                       ('enet', ElasticNet(alpha=1))])),
                   ('gbm1', GradientBoostingRegressor(
                       n_estimators=10, max_depth=2, min_samples_leaf=20)),
                   ('gbm2', GradientBoostingRegressor(
                       n_estimators=50, max_depth=2, min_samples_leaf=20)),
                   ('gbm3', GradientBoostingRegressor(
                       n_estimators=10, max_depth=3, min_samples_leaf=20)),
                   ('gbm4', GradientBoostingRegressor(
                       n_estimators=50, max_depth=3, min_samples_leaf=20)),
                   ('gbm5', GradientBoostingRegressor(learning_rate=.01,
                                                      n_estimators=100, max_depth=2, min_samples_leaf=20)),
                   ('gbm6', GradientBoostingRegressor(learning_rate=.01,
                                                      n_estimators=100, max_depth=2, min_samples_leaf=20)),
                   ('gbm7', GradientBoostingRegressor(learning_rate=.01,
                                                      n_estimators=100, max_depth=3, min_samples_leaf=20)),
                   ('gbm8', GradientBoostingRegressor(learning_rate=.01,
                                                      n_estimators=100, max_depth=3, min_samples_leaf=20)),
                   ('svr1', SVR(kernel='rbf')),
                   ('svr2', SVR(kernel='poly', degree=2)),
                   ('svr3', SVR(kernel='poly', degree=3)),
                   ('svr3', SVR(kernel='poly', degree=4)),
                   ('ada1', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), n_estimators=10, loss='linear')),
                   ('ada2', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), n_estimators=10, loss='square')),
                   ('ada3', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), learning_rate=.1, n_estimators=10, loss='linear')),
                   ('ada4', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), learning_rate=.1, n_estimators=50, loss='linear')),
                   ('ada5', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), learning_rate=.1, n_estimators=50, loss='square')),
                   ('ada6', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=1, min_samples_leaf=20), learning_rate=.1, n_estimators=10, loss='square')),
                   ('ada7', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=3, min_samples_leaf=20), learning_rate=.1, n_estimators=50, loss='square')),
                   ('ada8', AdaBoostRegressor(DecisionTreeRegressor(
                       max_depth=3, min_samples_leaf=20), learning_rate=.1, n_estimators=10, loss='square')),
                   ]
    meta = DRLearner(reg, clf, cate_models).fit(Z, D, Y)

    ####################################################
    # Evaluate on validation and construct ensemble
    ####################################################
    ensemble = {}
    for name, nuisance_mode in [('train', 'from_train'),
                                ('val', 'train_on_val'),
                                ('split', 'split_on_val'),
                                ('cfit', 'cfit_on_val'),
                                ('3way', '3way')]:

        if name not in ensemble_methods:
            continue

        print(f'Fitting ensemble {name} on dval')
        ensemble[name] = Ensemble(
            meta=meta, nuisance_mode=nuisance_mode, model_select=model_select)
        ensemble[name].fit(Zval, Dval, Yval)

    if g0 is not None:
        print('Fitting oracle ensemble on dval')
        ensemble['oracle'] = Ensemble(
            meta=meta, nuisance_mode='oracle', g0=g0, g1=g1, mu=mu)
        ensemble['oracle'].fit(Zval, Dval, Yval)

    ####################################################
    # Final evaluation on test set
    ####################################################
    print('Evaluating all models on dtest')
    F = meta.predict(Ztest)
    names = meta.cate_names()
    mses = {}
    cates = {}
    nuisance_metrics = {}
    subsample = np.sort(np.random.choice(Ztest.shape[0], 100))
    for it, name in enumerate(names):
        mses[name] = mse(cate(Ztest), F[:, it])
        cates[name] = F[subsample, it]
    cates['True'] = cate(Ztest[subsample])
    cates['Ztest'] = Ztest[subsample]

    tUniform = F @ np.ones(F.shape[1]) / F.shape[1]
    mses['Uni'] = mse(cate(Ztest), tUniform)
    cates['Uni'] = tUniform[subsample]
    for name, ensemble_model in ensemble.items():
        tQ = ensemble_model.predictQ(Ztest)
        tBest = ensemble_model.predictBest(Ztest)
        tConvex = ensemble_model.predictConvex(Ztest)
        mses[f'Q{name}'] = mse(cate(Ztest), tQ)
        mses[f'Best{name}'] = mse(cate(Ztest), tBest)
        mses[f'Convex{name}'] = mse(cate(Ztest), tConvex)
        cates[f'Q{name}'] = tQ[subsample]
        cates[f'Best{name}'] = tBest[subsample]
        cates[f'Convex{name}'] = tConvex[subsample]
        if g0 is not None:
            nuisance_subsample = np.sort(
                np.random.choice(ensemble_model.Zval.shape[0], 100))
            nuisance_metrics[name] = {'g0mse': mse(g0(ensemble_model.Zval), ensemble_model.g0preds),
                                      'g1mse': mse(g1(ensemble_model.Zval), ensemble_model.g1preds),
                                      'mumse': mse(mu(ensemble_model.Zval), ensemble_model.mupreds),
                                      'g0': ensemble_model.g0preds[nuisance_subsample],
                                      'g1': ensemble_model.g1preds[nuisance_subsample],
                                      'mu': ensemble_model.mupreds[nuisance_subsample],
                                      'g0true': g0(ensemble_model.Zval[nuisance_subsample]),
                                      'g1true': g1(ensemble_model.Zval[nuisance_subsample]),
                                      'mutrue': mu(ensemble_model.Zval[nuisance_subsample]),
                                      'Zval': ensemble_model.Zval[nuisance_subsample]}

    return mses, cates, nuisance_metrics


def experiment(dgp, *, n=None, semi_synth=False, simple_synth=False,
               scale=1, true_f=None, max_depth=3, random_state=123,
               reg=xgb_reg,
               clf=xgb_clf,
               wreg=xgb_wreg,
               model_select=False,
               ensemble_methods=['train', 'val', 'split', 'cfit', '3way'],
               prior_dict=None,
               beta=None, nu=0.5):

    np.random.seed(random_state)
    dtrain, dval, dtest, cate, g0, g1, mu = gen_data(dgp, n=n, semi_synth=semi_synth, simple_synth=simple_synth,
                                                     scale=scale, true_f=true_f, max_depth=max_depth,
                                                     random_state=random_state)
    Z, D, Y = dtrain
    Zval, Dval, Yval = dval
    Ztest, _, _ = dtest

    #############################################
    # Train on training set all meta-learners
    #############################################
    print('Fitting meta learners on dtrain')
    meta = MetaLearners(reg, clf, wreg).fit(Z, D, Y)
    
    prior = None if prior_dict is None else meta.get_prior(prior_dict)

    ####################################################
    # Evaluate on validation and construct ensemble
    ####################################################
    ensemble = {}
    for name, nuisance_mode in [('train', 'from_train'),
                                ('val', 'train_on_val'),
                                ('split', 'split_on_val'),
                                ('cfit', 'cfit_on_val'),
                                ('3way', '3way')]:

        if name not in ensemble_methods:
            continue

        print(f'Fitting ensemble {name} on dval')
        ensemble[name] = Ensemble(
            meta=meta, nuisance_mode=nuisance_mode, model_select=model_select,
            prior=prior, beta=beta, nu=nu
        )
        ensemble[name].fit(Zval, Dval, Yval)

    if g0 is not None:
        print('Fitting oracle ensemble on dval')
        ensemble['oracle'] = Ensemble(
            meta=meta, nuisance_mode='oracle', g0=g0, g1=g1, mu=mu,
            prior=prior, beta=beta, nu=nu
        )
        ensemble['oracle'].fit(Zval, Dval, Yval)

    ####################################################
    # Meta learners on all dtrain and dval
    ####################################################
    print('Fitting meta learners on dtrain union dval')
    meta_all = MetaLearners(reg, clf, wreg).fit(
        np.vstack([Z, Zval]), np.concatenate((D, Dval)), np.concatenate((Y, Yval)))

    ####################################################
    # Final evaluation on test set
    ####################################################
    print('Evaluating all models on dtest')
    F = meta.predict(Ztest)
    subsample = np.sort(np.random.choice(Ztest.shape[0], 100))
    mses = {'T': mse(cate(Ztest), F[:, 0]),
            'S': mse(cate(Ztest), F[:, 1]),
            'IPS': mse(cate(Ztest), F[:, 2]),
            'DR': mse(cate(Ztest), F[:, 3]),
            'R': mse(cate(Ztest), F[:, 4]),
            'X': mse(cate(Ztest), F[:, 5]),
            'DRX': mse(cate(Ztest), F[:, 6]),
            'DAX': mse(cate(Ztest), F[:, 7])}
    cates = {'T': F[subsample, 0],
             'S': F[subsample, 1],
             'IPS': F[subsample, 2],
             'DR': F[subsample, 3],
             'R': F[subsample, 4],
             'X': F[subsample, 5],
             'DRX': F[subsample, 6],
             'DAX': F[subsample, 7],
             'True': cate(Ztest[subsample]),
             'Ztest': Ztest[subsample]}
    nuisance_metrics = {}
    
    tUniform = F @ np.ones(F.shape[1]) / F.shape[1]
    mses['Uni'] = mse(cate(Ztest), tUniform)
    cates['Uni'] = tUniform[subsample]
    for name, ensemble_model in ensemble.items():
        tQ = ensemble_model.predictQ(Ztest)
        tBest = ensemble_model.predictBest(Ztest)
        tConvex = ensemble_model.predictConvex(Ztest)
        mses[f'Q{name}'] = mse(cate(Ztest), tQ)
        mses[f'Best{name}'] = mse(cate(Ztest), tBest)
        mses[f'Convex{name}'] = mse(cate(Ztest), tConvex)
        cates[f'Q{name}'] = tQ[subsample]
        cates[f'Best{name}'] = tBest[subsample]
        cates[f'Convex{name}'] = tConvex[subsample]
        if g0 is not None:
            nuisance_subsample = np.sort(
                np.random.choice(ensemble_model.Zval.shape[0], 100))
            nuisance_metrics[name] = {'g0mse': mse(g0(ensemble_model.Zval), ensemble_model.g0preds),
                                      'g1mse': mse(g1(ensemble_model.Zval), ensemble_model.g1preds),
                                      'mumse': mse(mu(ensemble_model.Zval), ensemble_model.mupreds),
                                      'g0': ensemble_model.g0preds[nuisance_subsample],
                                      'g1': ensemble_model.g1preds[nuisance_subsample],
                                      'mu': ensemble_model.mupreds[nuisance_subsample],
                                      'g0true': g0(ensemble_model.Zval[nuisance_subsample]),
                                      'g1true': g1(ensemble_model.Zval[nuisance_subsample]),
                                      'mutrue': mu(ensemble_model.Zval[nuisance_subsample]),
                                      'Zval': ensemble_model.Zval[nuisance_subsample]}

    F = meta_all.predict(Ztest)
    mses = {**mses,
            'Tall': mse(cate(Ztest), F[:, 0]),
            'Sall': mse(cate(Ztest), F[:, 1]),
            'IPSall': mse(cate(Ztest), F[:, 2]),
            'DRall': mse(cate(Ztest), F[:, 3]),
            'Rall': mse(cate(Ztest), F[:, 4]),
            'Xall': mse(cate(Ztest), F[:, 5]),
            'DRXall': mse(cate(Ztest), F[:, 6]),
            'DAXall': mse(cate(Ztest), F[:, 7])}
    cates = {**cates,
             'Tall': F[subsample, 0],
             'Sall': F[subsample, 1],
             'IPSall': F[subsample, 2],
             'DRall': F[subsample, 3],
             'Rall': F[subsample, 4],
             'Xall': F[subsample, 5],
             'DRXall': F[subsample, 6],
             'DAXall': F[subsample, 7]}

    return mses, cates, nuisance_metrics
