import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.datasets import load_breast_cancer, make_classification
from xgboost import XGBClassifier
from vqc import VQC

_HERE = os.path.dirname(os.path.abspath(__file__))
_UCI, _MIRROR = os.path.join(_HERE, "..", "data", "uci"), os.path.join(_HERE, "..", "data", "mirror")
_NEED = ("ionosphere.csv", "sonar.csv", "transfusion.csv")
DS_DIR = os.environ.get("VQC_DS_DIR") or (_UCI if all(os.path.exists(os.path.join(_UCI, f)) for f in _NEED) else _MIRROR)
DS_DIR = DS_DIR.rstrip("/\\") + os.sep
DATA_SOURCE = "uci" if os.path.abspath(DS_DIR).rstrip(os.sep) == os.path.abspath(_UCI) else ("mirror" if os.path.abspath(DS_DIR).rstrip(os.sep) == os.path.abspath(_MIRROR) else DS_DIR)
NJOBS = int(os.environ.get("VQC_JOBS", min(4, os.cpu_count() or 2)))      # worker processes
NSEEDS_ENV = int(os.environ.get("VQC_NSEEDS", 30))                         # 30 = paper setting; use 2 for a smoke test

def load_datasets():
    def csv(f):
        df = pd.read_csv(DS_DIR + f, header=None)
        return df.iloc[:, :-1].values.astype(float), df.iloc[:, -1].values
    data = {}
    b = load_breast_cancer(); data["WDBC"] = (b.data, b.target)
    X, y = csv("ionosphere.csv"); data["Ionosphere"] = (np.delete(X, 1, 1), (y == "g").astype(int))
    X, y = csv("sonar.csv"); data["Sonar"] = (X, (y == "M").astype(int))
    Xc, yc = make_classification(600, 30, n_informative=4, n_redundant=0, n_clusters_per_class=2,
                                 flip_y=0.05, random_state=0)
    data["Synthetic"] = (Xc, yc)
    X, y = csv("transfusion.csv"); data["Transfusion"] = (X, y.astype(int))   # 4 native features
    return data

def preprocess(Xtr, Xte, Xextra=None):
    """Standardize -> PCA(4) -> rescale to [-pi,pi]; every step fitted on the training part only."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if A.shape[1] > 4:                      # native 4-feature datasets are used as they are (no PCA)
        pca = PCA(4, random_state=0).fit(A)
        A, B = pca.transform(A), pca.transform(B)
    lo, hi = A.min(0), A.max(0)
    f = lambda Z: np.clip(2 * np.pi * (Z - lo) / (hi - lo) - np.pi, -np.pi, np.pi)
    return f(A), f(B)

def auc(y, s):
    return roc_auc_score(y, s)

def make_models(seed, lr, epochs, vqc_kw=None):
    vqc_kw = vqc_kw or {}
    return {
        "VQC": lambda: VQC(lr=lr, epochs=epochs, seed=seed, **vqc_kw),
        "LR": lambda: LogisticRegression(max_iter=2000),
        "SVM": lambda: SVC(probability=True, random_state=seed),
        "RF": lambda: RandomForestClassifier(200, random_state=seed, n_jobs=1),
        "XGB": lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                     random_state=seed, verbosity=0, n_jobs=1),
        "MLP": lambda: MLPClassifier((3,), activation="tanh", solver="adam", batch_size=10**6,
                                     learning_rate_init=lr, max_iter=epochs, tol=0,
                                     n_iter_no_change=10**6, random_state=seed),
    }

def scores(m, X):
    if hasattr(m, "predict_proba1"):
        d = m.decision_function(X); return d, m.predict_proba1(X)
    p = m.predict_proba(X)[:, 1]
    d = m.decision_function(X) if hasattr(m, "decision_function") else p
    return d, p

def evaluate(mk, Atr, ytr, Ate, yte):
    m = mk(); m.fit(Atr, ytr)
    dtr, ptr = scores(m, Atr); dte, pte = scores(m, Ate)
    return dict(auc_tr=auc(ytr, dtr), auc_te=auc(yte, dte),
                brier_te=brier_score_loss(yte, pte), brier_tr=brier_score_loss(ytr, ptr))

# ---- representation variants for H1 -------------------------------------------------
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_variant(Xtr, Xte, ytr, kind):
    """kind: 'pca4' (E1 default), 'anova4' (top-4 features by ANOVA F on the training part),
    'all' (standardized, no reduction; classical models only). Every step is fitted on the training part only."""
    if kind == "pca4":
        return preprocess(Xtr, Xte)
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    if kind == "all":
        return A, B
    if kind == "anova4":
        sel = SelectKBest(f_classif, k=4).fit(A, ytr); A, B = sel.transform(A), sel.transform(B)
        lo, hi = A.min(0), A.max(0)
        f = lambda Z: np.clip(2 * np.pi * (Z - lo) / (hi - lo) - np.pi, -np.pi, np.pi)
        return f(A), f(B)
    raise ValueError(kind)
