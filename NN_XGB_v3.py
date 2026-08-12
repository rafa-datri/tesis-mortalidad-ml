# ==========================================================
# Unified pipeline (SMOTE): XGBoost + Neural Network (MLP)
# Umbral por bin + Calibración Isotónica + WH + Monotonía
# qx con persona-años (Ortega) vs Tabla (IC Delta Poisson)
# ==========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
random.seed(123); np.random.seed(123)

# ---- Flags ------------------------------------------------
RUN_XGB = True
RUN_NN  = True

# ---- Config ----------------------------------------------
MAIN_PATH = r"C:\Users\Rafael\Desktop\UBA\Tesis_Actuarial\MET_DATA\_CODE"
OUTPUT_DIR = os.path.join(MAIN_PATH, "OUTPUT_MODEL_NN_XGB_final")
INPUT_FILE = "DF_TOTAL_CB.csv"
MORT_TABLE_FILE = "MORT_TABLE_CB_H_2020.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bins de edad (qx a 5 años)
BIN_WIDTH   = 5
AGE_MIN     = 50
AGE_MAX     = 105
AGE_BINS    = list(range(AGE_MIN, AGE_MAX + BIN_WIDTH, BIN_WIDTH))
AGE_LABELS  = [f"{AGE_BINS[i]}-{AGE_BINS[i+1]-1}" for i in range(len(AGE_BINS)-1)]

# Tiempo medio vivido por los que fallecen, como fracción del bin
K_DECEASED = 0.5   

# SEX FILTER: "M" (solo hombres), "F" (solo mujeres), "ALL" (ambos; usa BEN_SEXO_F como predictor)
SEX_FILTER = "M"
OBJ_UMBRAL = "COUNT_MATCH"     # "COUNT_MATCH", "F1", "Recall"
USE_WHITTAKER = True

# ---- Common imports --------------------------------------
from scipy.stats import norm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.sparse.linalg import spsolve
from imblearn.over_sampling import SMOTE

# ================== Utilities =============================
def ci_poisson_delta_from_py(d, E, n, alpha=0.05):
    """IC Delta para q = 1 - exp(-n*m) con d~Poisson(E*m)."""
    d = np.asarray(d, float)
    E = np.maximum(np.asarray(E, float), 1e-9)
    mhat = d / E
    qhat = 1.0 - np.exp(-n * mhat)
    se   = n * (1.0 - qhat) * np.sqrt(np.maximum(mhat / E, 0.0))
    z    = norm.ppf(1 - alpha/2)
    low  = np.clip(qhat - z*se, 0.0, 1.0)
    high = np.clip(qhat + z*se, 0.0, 1.0)
    return qhat, low, high

def whittaker_henderson(y, w, lmbd):
    y = np.asarray(y, dtype=float); w = np.asarray(w, dtype=float)
    m = len(y)
    W = sparse.diags(w, 0, shape=(m, m), format="csc")
    E = sparse.eye(m, format="csc")
    D1 = E[1:] - E[:-1]
    D2 = D1[1:] - D1[:-1]
    Z  = W + lmbd * (D2.T @ D2)
    rhs = W @ y
    return spsolve(Z, rhs)

def mae_weighted(y_true, y_pred, weights):
    return np.average(np.abs(np.asarray(y_true) - np.asarray(y_pred)),
                      weights=np.asarray(weights))

def make_bins(series):
    return pd.cut(series, bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)

def enforce_monotone_non_decreasing(values, weights):
    x = np.arange(len(values))
    ir = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    fitted = ir.fit_transform(x, np.asarray(values, float),
                              sample_weight=np.asarray(weights, float))
    return np.clip(fitted, 0.0, 1.0)

def select_threshold_by_bin(p_valid_cal, y_valid, valid_bins, obj=OBJ_UMBRAL):
    thr_grid = np.arange(0.01, 0.99, 0.01)
    thr_by_bin = {}
    for bin_label in AGE_LABELS:
        mask = (valid_bins == bin_label)
        if mask.sum() == 0:
            thr_by_bin[bin_label] = 0.5
            continue
        y_b = y_valid[mask]; p_b = p_valid_cal[mask]
        if obj == "F1":
            best_thr, best_val = 0.5, -1
            for t in thr_grid:
                val = f1_score(y_b, (p_b >= t).astype(int), zero_division=0)
                if val > best_val: best_val, best_thr = val, t
            thr_by_bin[bin_label] = float(best_thr)
        elif obj == "Recall":
            best_thr, best_val = 0.5, -1
            for t in thr_grid:
                val = recall_score(y_b, (p_b >= t).astype(int), zero_division=0)
                if val > best_val: best_val, best_thr = val, t
            thr_by_bin[bin_label] = float(best_thr)
        else:  # COUNT_MATCH
            qx_real_b = y_b.mean()
            best_thr, best_err = 0.5, 1e9
            for t in thr_grid:
                err = abs((p_b >= t).mean() - qx_real_b)
                if err < best_err: best_err, best_thr = err, t
            thr_by_bin[bin_label] = float(best_thr)
    return thr_by_bin

# ================== Load Data =============================
def load_data(file_name, sep=";"):
    df = pd.read_csv(os.path.join(MAIN_PATH, file_name), sep=sep)
    df.columns = df.columns.str.strip()
    df["muere"] = df["muere"].astype(int)
    if "Edad_Inicio" in df.columns and df["Edad_Inicio"].dtype == object:
        df["Edad_Inicio"] = df["Edad_Inicio"].str.replace(",", ".", regex=False)
        df["Edad_Inicio"] = pd.to_numeric(df["Edad_Inicio"], errors="coerce")
    return df

df = load_data(INPUT_FILE)

# ====== FILTRO POR SEXO según SEX_FILTER ======
if "BEN_SEXO_F" not in df.columns:
    raise KeyError("No se encontró la columna 'BEN_SEXO_F' en el dataset.")

if SEX_FILTER not in {"M", "F", "ALL"}:
    raise ValueError("SEX_FILTER debe ser 'M', 'F' o 'ALL'.")

before_rows = len(df)
if SEX_FILTER == "M":
    df = df[df["BEN_SEXO_F"] != 1].copy()
    df.drop(columns=["BEN_SEXO_F"], inplace=True)
    print(f"[Sexo] Solo HOMBRES: {before_rows} → {len(df)} filas. Columna BEN_SEXO_F eliminada.")
elif SEX_FILTER == "F":
    df = df[df["BEN_SEXO_F"] == 1].copy()
    df.drop(columns=["BEN_SEXO_F"], inplace=True)
    print(f"[Sexo] Solo MUJERES: {before_rows} → {len(df)} filas. Columna BEN_SEXO_F eliminada.")
else:
    print(f"[Sexo] HOMBRES+MUJERES: {before_rows} → {len(df)} filas. BEN_SEXO_F se mantiene como predictor.")

# X / y
X = df.drop(columns=["muere","BEN_NUPOL","BEN_RUT","DEATH_YEAR"], errors="ignore")
y = df["muere"].values

# Split 60/20/20
from sklearn.model_selection import train_test_split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=123)
X_train, X_valid, y_train, y_valid = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=123)

# Bins (need raw Edad_Inicio)
valid_bins = make_bins(X_valid["Edad_Inicio"])
test_bins  = make_bins(X_test["Edad_Inicio"])

# ================== SMOTE on TRAIN only ==================
print("Before SMOTE:", np.bincount(y_train))
sm = SMOTE(random_state=123, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print("After  SMOTE:", np.bincount(y_train_sm))

# =============== Model 1: Neural Network ==================
nn_results = None
if RUN_NN:
    import tensorflow as tf
    tf.random.set_seed(123)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    scaler_nn = StandardScaler()
    X_train_sm_s = scaler_nn.fit_transform(X_train_sm)
    X_valid_s    = scaler_nn.transform(X_valid)
    X_test_s     = scaler_nn.transform(X_test)
    joblib.dump(scaler_nn, os.path.join(OUTPUT_DIR, "scaler_nn.pkl"))

    def build_nn(input_dim, lr=0.001, dropout=0.30):
        model = Sequential([
            Dense(64, activation="relu", input_dim=input_dim),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dropout(dropout),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=["AUC"])
        return model

    nn = build_nn(X_train_sm_s.shape[1], lr=0.001, dropout=0.30)
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    nn.fit(X_train_sm_s, y_train_sm,
           validation_data=(X_valid_s, y_valid),
           epochs=200, batch_size=256,
           callbacks=[es], verbose=1)

    nn.save(os.path.join(OUTPUT_DIR,"NN_model.h5"))

    p_valid = nn.predict(X_valid_s).ravel()
    p_test  = nn.predict(X_test_s).ravel()

    iso_nn = IsotonicRegression(out_of_bounds="clip")
    iso_nn.fit(p_valid, y_valid)
    p_valid_cal_nn = iso_nn.transform(p_valid)
    p_test_cal_nn  = iso_nn.transform(p_test)

    thr_by_bin_nn = select_threshold_by_bin(p_valid_cal_nn, y_valid, valid_bins, OBJ_UMBRAL)
    thr_global_nn = float(np.median(list(thr_by_bin_nn.values())))
    thr_map_nn    = pd.Series(test_bins, index=X_test.index).map(thr_by_bin_nn)
    thr_for_test_nn = pd.to_numeric(thr_map_nn, errors="coerce").fillna(thr_global_nn).values
    y_pred_bin_nn = (p_test_cal_nn >= thr_for_test_nn).astype(int)

    # Métricas
    acc = (y_pred_bin_nn==y_test).mean()
    prec = precision_score(y_test, y_pred_bin_nn, zero_division=0)
    rec  = recall_score(y_test, y_pred_bin_nn, zero_division=0)
    f1   = f1_score(y_test, y_pred_bin_nn, zero_division=0)
    auc  = roc_auc_score(y_test, p_test_cal_nn)
    cm_nn = confusion_matrix(y_test, y_pred_bin_nn)

    nn_results = {
        "p_test_cal": p_test_cal_nn,
        "y_pred_bin": y_pred_bin_nn,
        "metrics": {"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,"AUC":auc,"CM":cm_nn},
        "thr_by_bin": thr_by_bin_nn
    }

# =============== Model 2: XGBoost =========================
xgb_results = None
if RUN_XGB:
    import xgboost as xgb
    from xgboost import XGBClassifier

    mono = [1 if c=="Edad_Inicio" else 0 for c in X.columns]
    monotone_constraints = "(" + ",".join(map(str, mono)) + ")"

    base = XGBClassifier(
        random_state=123, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", max_delta_step=1,
        monotone_constraints=monotone_constraints
        # sin scale_pos_weight al usar SMOTE
    )
    param_space = {
        "n_estimators":[400,700,1000],
        "learning_rate":[0.02,0.05,0.10],
        "max_depth":[3,5,7],
        "min_child_weight":[1,5,10],
        "gamma":[0,0.1,0.3],
        "subsample":[0.7,0.85,1.0],
        "colsample_bytree":[0.7,0.85,1.0],
        "reg_alpha":[0,0.1,0.5],
        "reg_lambda":[1,1.5,2]
    }
    search = RandomizedSearchCV(
        base, param_space, scoring="average_precision", n_iter=30, cv=3,
        random_state=123, verbose=1, n_jobs=-1
    )
    search.fit(X_train_sm, y_train_sm)
    best_params = search.best_params_
    joblib.dump(best_params, os.path.join(OUTPUT_DIR,"xgb_best_params.pkl"))

    dtrain = xgb.DMatrix(X_train_sm, label=y_train_sm)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "objective":"binary:logistic","eval_metric":"logloss",
        "tree_method":"hist","max_delta_step":1,
        "monotone_constraints":monotone_constraints,"random_state":123,
        **{k:v for k,v in best_params.items() if k!="n_estimators"}
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=best_params["n_estimators"],
        evals=[(dtrain,"train"),(dvalid,"valid")], early_stopping_rounds=100, verbose_eval=False
    )
    booster.save_model(os.path.join(OUTPUT_DIR,"XGB_booster.json"))

    p_valid = booster.predict(dvalid, iteration_range=(0, booster.best_iteration+1))
    p_test  = booster.predict(dtest,  iteration_range=(0, booster.best_iteration+1))

    iso_xgb = IsotonicRegression(out_of_bounds="clip")
    iso_xgb.fit(p_valid, y_valid)
    p_valid_cal_xgb = iso_xgb.transform(p_valid)
    p_test_cal_xgb  = iso_xgb.transform(p_test)

    thr_by_bin_xgb = select_threshold_by_bin(p_valid_cal_xgb, y_valid, valid_bins, OBJ_UMBRAL)
    thr_global_xgb = float(np.median(list(thr_by_bin_xgb.values())))
    thr_map_xgb    = pd.Series(test_bins, index=X_test.index).map(thr_by_bin_xgb)
    thr_for_test_xgb = pd.to_numeric(thr_map_xgb, errors="coerce").fillna(thr_global_xgb).values
    y_pred_bin_xgb = (p_test_cal_xgb >= thr_for_test_xgb).astype(int)

    # Métricas
    acc = (y_pred_bin_xgb==y_test).mean()
    prec = precision_score(y_test, y_pred_bin_xgb, zero_division=0)
    rec  = recall_score(y_test, y_pred_bin_xgb, zero_division=0)
    f1   = f1_score(y_test, y_pred_bin_xgb, zero_division=0)
    auc  = roc_auc_score(y_test, p_test_cal_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_bin_xgb)

    xgb_results = {
        "p_test_cal": p_test_cal_xgb,
        "y_pred_bin": y_pred_bin_xgb,
        "metrics": {"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,"AUC":auc,"CM":cm_xgb},
        "thr_by_bin": thr_by_bin_xgb
    }

# =============== Aggregations (common + per model) =================
# Base por bin con muertes y expuestos
df_base = pd.DataFrame({"AGE_BIN": test_bins, "muere": y_test}).dropna(subset=["AGE_BIN"])
base_agg = df_base.groupby("AGE_BIN", observed=True).agg(
    muertes=("muere","sum"),
    expuestos=("muere","count"),
).reset_index()
base_agg["AGE_BIN"] = base_agg["AGE_BIN"].astype(str)

# muertes_pred por modelo (separadas)
if RUN_NN:
    df_nn = pd.DataFrame({"AGE_BIN": test_bins, "pred_nn": nn_results["y_pred_bin"]}).dropna(subset=["AGE_BIN"])
    agg_nn = df_nn.groupby("AGE_BIN", observed=True)["pred_nn"].sum().rename("muertes_pred_nn").reset_index()
    base_agg = base_agg.merge(agg_nn, on="AGE_BIN", how="left")
if RUN_XGB:
    df_xgb = pd.DataFrame({"AGE_BIN": test_bins, "pred_xgb": xgb_results["y_pred_bin"]}).dropna(subset=["AGE_BIN"])
    agg_xgb = df_xgb.groupby("AGE_BIN", observed=True)["pred_xgb"].sum().rename("muertes_pred_xgb").reset_index()
    base_agg = base_agg.merge(agg_xgb, on="AGE_BIN", how="left")

# Persona-años (Ortega): L = Sum(l_x - (1-k)*d_x) .- La suma será entre las edades del tramo
base_agg["py_ortega"] = (base_agg["expuestos"] - (1.0 - K_DECEASED) * base_agg["muertes"])
base_agg["py_ortega"] = base_agg["py_ortega"].clip(lower=1e-9)

# qx real con Ortega
m_real = base_agg["muertes"] / base_agg["py_ortega"]
base_agg["qx_real"] = 1.0 - np.exp(-BIN_WIDTH * m_real)

# qx modelo(s) con el MISMO denominador de persona-años
if RUN_NN:
    m_nn = base_agg["muertes_pred_nn"] / base_agg["py_ortega"]
    base_agg["qx_model_nn"] = 1.0 - np.exp(-BIN_WIDTH * m_nn)
if RUN_XGB:
    m_xgb = base_agg["muertes_pred_xgb"] / base_agg["py_ortega"]
    base_agg["qx_model_xgb"] = 1.0 - np.exp(-BIN_WIDTH * m_xgb)

# IC Delta Poisson–exposición para qx_real
_, low, high = ci_poisson_delta_from_py(base_agg["muertes"], base_agg["py_ortega"], BIN_WIDTH, alpha=0.05)
base_agg["qx_real_lower"] = low; base_agg["qx_real_upper"] = high

# =============== Mortality Table qx by bin =================
mort = pd.read_csv(os.path.join(MAIN_PATH, MORT_TABLE_FILE), sep=";")
mort.columns = mort.columns.str.strip()
mort = mort.rename(columns={mort.columns[0]:"Edad", mort.columns[1]:"qx_tabla"})
if mort["Edad"].dtype==object:
    mort["Edad"]=mort["Edad"].str.replace(",",".",regex=False)
    mort["Edad"]=pd.to_numeric(mort["Edad"], errors="coerce")
mort["qx_tabla"]=mort["qx_tabla"].astype(str).str.replace(",",".",regex=False)
mort["qx_tabla"]=pd.to_numeric(mort["qx_tabla"], errors="coerce")
mort = mort.sort_values("Edad").dropna(subset=["Edad"]).reset_index(drop=True)
age_min=int(mort["Edad"].min()); age_max=int(mort["Edad"].max())
mort = (mort.set_index("Edad").reindex(range(age_min,age_max+1)).rename_axis("Edad").reset_index())
RADIX=100000.0; surv=1-mort["qx_tabla"].fillna(0)
mort["l_x"]=RADIX*surv.shift(fill_value=1).cumprod()
rows=[]
for lo,hi in zip(AGE_BINS[:-1], AGE_BINS[1:]):
    lx_lo=mort.loc[mort["Edad"]==lo,"l_x"]; lx_hi=mort.loc[mort["Edad"]==hi,"l_x"]
    q_bin=np.nan if (lx_lo.empty or lx_hi.empty or pd.isna(lx_lo.iloc[0]) or pd.isna(lx_hi.iloc[0])) else 1.0-(lx_hi.iloc[0]/lx_lo.iloc[0])
    rows.append({"AGE_BIN":f"{lo}-{hi-1}","qx_tabla_lx":q_bin})
qx_tabla_bin_lx=pd.DataFrame(rows)
qx_tabla_bin_lx["AGE_BIN"]=qx_tabla_bin_lx["AGE_BIN"].astype(str)

# Merge tabla
df_final = base_agg.merge(qx_tabla_bin_lx, on="AGE_BIN", how="left")
df_final["AGE_LO"] = df_final["AGE_BIN"].str.split("-").str[0].astype(int)
df_final = df_final.sort_values("AGE_LO").reset_index(drop=True)

# =============== WH & Monotonía (pesos = persona-años) ===============
if RUN_NN and "qx_model_nn" in df_final:
    df_final["qx_model_nn_mon"] = enforce_monotone_non_decreasing(
        df_final["qx_model_nn"].values, df_final["py_ortega"].values
    )
    if USE_WHITTAKER:
        LAMBDA_GRID = [10,30,100,300,1000,3000,10000]
        y_wh = df_final["qx_model_nn"].values; w_wh = df_final["py_ortega"].values
        best_lmbd_nn, best_score_nn = None, np.inf
        for lmbd in LAMBDA_GRID:
            b_try = whittaker_henderson(y_wh, w_wh, lmbd)
            score = mae_weighted(df_final["qx_real"].values, b_try, w_wh)
            if score < best_score_nn: best_score_nn, best_lmbd_nn = score, lmbd
        df_final["qx_model_nn_wh"] = np.clip(whittaker_henderson(y_wh, w_wh, best_lmbd_nn), 0, 1)
        df_final["qx_model_nn_wh_mon"] = enforce_monotone_non_decreasing(
            df_final["qx_model_nn_wh"].values, df_final["py_ortega"].values
        )
    else:
        best_lmbd_nn = None

if RUN_XGB and "qx_model_xgb" in df_final:
    df_final["qx_model_xgb_mon"] = enforce_monotone_non_decreasing(
        df_final["qx_model_xgb"].values, df_final["py_ortega"].values
    )
    if USE_WHITTAKER:
        LAMBDA_GRID = [10,30,100,300,1000,3000,10000]
        y_wh = df_final["qx_model_xgb"].values; w_wh = df_final["py_ortega"].values
        best_lmbd_xgb, best_score_xgb = None, np.inf
        for lmbd in LAMBDA_GRID:
            b_try = whittaker_henderson(y_wh, w_wh, lmbd)
            score = mae_weighted(df_final["qx_real"].values, b_try, w_wh)
            if score < best_score_xgb: best_score_xgb, best_lmbd_xgb = score, lmbd
        df_final["qx_model_xgb_wh"] = np.clip(whittaker_henderson(y_wh, w_wh, best_lmbd_xgb), 0, 1)
        df_final["qx_model_xgb_wh_mon"] = enforce_monotone_non_decreasing(
            df_final["qx_model_xgb_wh"].values, df_final["py_ortega"].values
        )
    else:
        best_lmbd_xgb = None

# ================== MAE (per column) ======================
def mae_cols(df, col):
    m  = (df[col] - df["qx_real"]).abs().mean()
    mw = mae_weighted(df["qx_real"], df[col], df["py_ortega"])   # pesos = persona-años
    return m, mw

summary_rows = []
if RUN_NN and "qx_model_nn" in df_final:
    for col, label in [
        ("qx_model_nn","NN"),
        ("qx_model_nn_mon","NN"),
    ] + ( [("qx_model_nn_wh","NN (WH)"), ("qx_model_nn_wh_mon","NN (WH)")] if USE_WHITTAKER and "qx_model_nn_wh" in df_final else [] ):
        m, mw = mae_cols(df_final, col); summary_rows += [
            {"métrica": f"MAE {label}", "valor": m, "detalle": "no ponderado"},
            {"métrica": f"MAE {label} (pond)", "valor": mw, "detalle": "ponderado"},
        ]

if RUN_XGB and "qx_model_xgb" in df_final:
    for col, label in [
        ("qx_model_xgb","XGB"),
        ("qx_model_xgb_mon","XGB"),
    ] + ( [("qx_model_xgb_wh","XGB (WH)"), ("qx_model_xgb_wh_mon","XGB (WH)")] if USE_WHITTAKER and "qx_model_xgb_wh" in df_final else [] ):
        m, mw = mae_cols(df_final, col); summary_rows += [
            {"métrica": f"MAE {label}", "valor": m, "detalle": "no ponderado"},
            {"métrica": f"MAE {label} (pond)", "valor": mw, "detalle": "ponderado"},
        ]

m_tabla, mw_tabla = mae_cols(df_final, "qx_tabla_lx")
summary_rows += [
    {"métrica":"MAE Tabla","valor":m_tabla,"detalle":"no ponderado"},
    {"métrica":"MAE Tabla (pond)","valor":mw_tabla,"detalle":"ponderado"}
]
if RUN_NN and USE_WHITTAKER and "qx_model_nn_wh" in df_final:
    summary_rows += [{"métrica":"λ WH óptimo NN","valor":float(best_lmbd_nn),"detalle":"grid search"}]
if RUN_XGB and USE_WHITTAKER and "qx_model_xgb_wh" in df_final:
    summary_rows += [{"métrica":"λ WH óptimo XGB","valor":float(best_lmbd_xgb),"detalle":"grid search"}]
df_resumen = pd.DataFrame(summary_rows)

# ================== Errors (for report) ====================
def add_abs_err(df, col, name):
    df[f"abs_error_{name}"] = (df[col] - df["qx_real"]).abs()

if RUN_NN and "qx_model_nn" in df_final:
    add_abs_err(df_final, "qx_model_nn", "nn")
    add_abs_err(df_final, "qx_model_nn_mon", "nn_mon")
    if USE_WHITTAKER and "qx_model_nn_wh" in df_final:
        add_abs_err(df_final, "qx_model_nn_wh", "nn_wh")
        add_abs_err(df_final, "qx_model_nn_wh_mon", "nn_wh_mon")
if RUN_XGB and "qx_model_xgb" in df_final:
    add_abs_err(df_final, "qx_model_xgb", "xgb")
    add_abs_err(df_final, "qx_model_xgb_mon", "xgb_mon")
    if USE_WHITTAKER and "qx_model_xgb_wh" in df_final:
        add_abs_err(df_final, "qx_model_xgb_wh", "xgb_wh")
        add_abs_err(df_final, "qx_model_xgb_wh_mon", "xgb_wh_mon")
add_abs_err(df_final, "qx_tabla_lx", "tabla")

# ================== Metrics & Info Sheets =================
metrics_rows = []
cm_blocks = []
if RUN_NN:
    m = nn_results["metrics"]
    metrics_rows += [
        {"model":"NN","metric":"Accuracy","value":m["Accuracy"]},
        {"model":"NN","metric":"Precision","value":m["Precision"]},
        {"model":"NN","metric":"Recall","value":m["Recall"]},
        {"model":"NN","metric":"F1","value":m["F1"]},
        {"model":"NN","metric":"AUC (calibrado)","value":m["AUC"]},
    ]
    cm_blocks.append(("NN", m["CM"]))
if RUN_XGB:
    m = xgb_results["metrics"]
    metrics_rows += [
        {"model":"XGB","metric":"Accuracy","value":m["Accuracy"]},
        {"model":"XGB","metric":"Precision","value":m["Precision"]},
        {"model":"XGB","metric":"Recall","value":m["Recall"]},
        {"model":"XGB","metric":"F1","value":m["F1"]},
        {"model":"XGB","metric":"AUC (calibrado)","value":m["AUC"]},
    ]
    cm_blocks.append(("XGB", m["CM"]))
df_metrics = pd.DataFrame(metrics_rows)
df_info = pd.DataFrame([
    {"setting":"OBJ_UMBRAL","value":OBJ_UMBRAL},
    {"setting":"BALANCEO","value":"SMOTE en TRAIN (sin class_weight/scale_pos_weight)"},
    {"setting":"SEX_FILTER","value":SEX_FILTER},
    {"setting":"K_DECEASED","value":K_DECEASED}
])

# ================== Excel Export ==========================
excel_path = os.path.join(OUTPUT_DIR, "qx_comp_SMOTE_Ortega.xlsx")
engine_excel = "xlsxwriter"
try:
    import xlsxwriter 
except Exception:
    engine_excel = "openpyxl"

cols = ["AGE_BIN","AGE_LO","muertes","expuestos","py_ortega"]
if RUN_NN:  cols.append("muertes_pred_nn")
if RUN_XGB: cols.append("muertes_pred_xgb")
cols += ["qx_real","qx_real_lower","qx_real_upper","qx_tabla_lx"]
if RUN_NN:
    cols += ["qx_model_nn","qx_model_nn_mon"]
    if USE_WHITTAKER and "qx_model_nn_wh" in df_final:
        cols += ["qx_model_nn_wh","qx_model_nn_wh_mon"]
if RUN_XGB:
    cols += ["qx_model_xgb","qx_model_xgb_mon"]
    if USE_WHITTAKER and "qx_model_xgb_wh" in df_final:
        cols += ["qx_model_xgb_wh","qx_model_xgb_wh_mon"]
cols += [c for c in df_final.columns if c.startswith("abs_error_")]

with pd.ExcelWriter(excel_path, engine=engine_excel) as writer:
    df_final[cols].to_excel(writer, sheet_name="qx_por_bin", index=False)
    df_resumen.to_excel(writer, sheet_name="resumen_mae", index=False)
    df_metrics.to_excel(writer, sheet_name="metrics_test", index=False)
    # append settings
    start_row = len(df_metrics) + 2
    df_info.to_excel(writer, sheet_name="metrics_test", index=False, startrow=start_row)

    # Confusion matrices
    r = start_row + len(df_info) + 2
    for name, cm in cm_blocks:
        title_df = pd.DataFrame({f"Matriz de confusión - {name}": [""]})
        title_df.to_excel(writer, sheet_name="metrics_test", index=False, header=True, startrow=r, startcol=0)
        r += 2
        pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["Real 0","Real 1"])\
          .to_excel(writer, sheet_name="metrics_test", startrow=r, startcol=0)
        r += 6

print("Excel generado en:", excel_path)

# ================== PLOTS + REAL (IC Delta desde persona-años) ==================
x = df_final["AGE_BIN"].values
_, low_ci, up_ci = ci_poisson_delta_from_py(
    d=df_final["muertes"].values,
    E=df_final["py_ortega"].values,
    n=BIN_WIDTH,
    alpha=0.05
)

def _decorate_and_save(title, outname):
    plt.plot(x, df_final["qx_real"], marker="o", linewidth=2, label="qₓ Real")
    plt.fill_between(x, low_ci, up_ci, alpha=0.2, label="IC 95% qₓ Real (Delta)")
    plt.xlabel("Age band"); plt.ylabel("qₓ")
    plt.title(title)
    plt.grid(True); plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=300); plt.close()

# 1) NN + NN (Whittaker) + Tabla
if RUN_NN and "qx_model_nn_mon" in df_final:
    plt.figure(figsize=(12, 6))
    plt.plot(x, df_final["qx_model_nn_mon"], marker="s", linewidth=2, label="qₓ NN")
    if USE_WHITTAKER and "qx_model_nn_wh_mon" in df_final:
        plt.plot(x, df_final["qx_model_nn_wh_mon"], marker="x", linewidth=2, label="qₓ NN (Whittaker)")
    plt.plot(x, df_final["qx_tabla_lx"], marker="d", linestyle="--", linewidth=2, label="qₓ Tabla")
    _decorate_and_save("qₓ — NN vs Whittaker vs Tabla", "plot_qx_nn_vs_table.png")

# 2) XGB + XGB (Whittaker) + Tabla
if RUN_XGB and "qx_model_xgb_mon" in df_final:
    plt.figure(figsize=(12, 6))
    plt.plot(x, df_final["qx_model_xgb_mon"], marker="^", linewidth=2, label="qₓ XGB")
    if USE_WHITTAKER and "qx_model_xgb_wh_mon" in df_final:
        plt.plot(x, df_final["qx_model_xgb_wh_mon"], marker="v", linewidth=2, label="qₓ XGB (Whittaker)")
    plt.plot(x, df_final["qx_tabla_lx"], marker="d", linestyle="--", linewidth=2, label="qₓ Tabla")
    _decorate_and_save("qₓ — XGB vs Whittaker vs Tabla", "plot_qx_xgb_vs_table.png")

# 3) NN (Wh) + XGB (Wh) + Tabla
if RUN_NN and RUN_XGB and USE_WHITTAKER \
   and "qx_model_nn_wh_mon" in df_final and "qx_model_xgb_wh_mon" in df_final:
    plt.figure(figsize=(12, 6))
    plt.plot(x, df_final["qx_model_nn_wh_mon"],  marker="o", linewidth=2, label="qₓ NN (Whittaker)")
    plt.plot(x, df_final["qx_model_xgb_wh_mon"], marker="^", linewidth=2, label="qₓ XGB (Whittaker)")
    plt.plot(x, df_final["qx_tabla_lx"],         marker="d", linestyle="--", linewidth=2, label="qₓ Tabla")
    _decorate_and_save("qₓ — Whittaker (NN vs XGB) vs Tabla", "plot_qx_wh_both_vs_table.png")

print("Saved qx plots:",
      "plot_qx_nn_vs_table.png" if RUN_NN else "",
      "plot_qx_xgb_vs_table.png" if RUN_XGB else "",
      "plot_qx_wh_both_vs_table.png" if (RUN_NN and RUN_XGB and USE_WHITTAKER) else "")

# ================== METRICS COMPARISON (Class 1) ==================
metrics_labels = ["Precision", "Recall", "F1"]
nn_vals = []
xgb_vals = []

if RUN_NN:
    nn_vals = [nn_results["metrics"]["Precision"], nn_results["metrics"]["Recall"], nn_results["metrics"]["F1"]]
if RUN_XGB:
    xgb_vals = [xgb_results["metrics"]["Precision"], xgb_results["metrics"]["Recall"], xgb_results["metrics"]["F1"]]

if RUN_NN or RUN_XGB:
    idx = np.arange(len(metrics_labels))
    width = 0.35 if (RUN_NN and RUN_XGB) else 0.6

    plt.figure(figsize=(8, 5))
    if RUN_NN:
        plt.bar(idx - (width/2 if RUN_XGB else 0), nn_vals, width, label="NN")
    if RUN_XGB:
        plt.bar(idx + (width/2 if RUN_NN else 0), xgb_vals, width, label="XGB")

    plt.xticks(idx, metrics_labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Métricas (clase positiva): NN vs XGB")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_class1_nn_vs_xgb.png"), dpi=300)
    plt.close()

print("Saved metrics plot:", "metrics_class1_nn_vs_xgb.png")

# ================== MAE ponderado (comparación) ==================
mae_bars_labels = []
mae_bars_values = []

if RUN_NN:
    if "qx_model_nn_mon" in df_final:
        _, mw = mae_cols(df_final, "qx_model_nn_mon"); mae_bars_labels += ["NN"]; mae_bars_values += [mw]
    if "qx_model_nn_wh" in df_final:
        _, mw = mae_cols(df_final, "qx_model_nn_wh"); mae_bars_labels += ["NN (WH)"]; mae_bars_values += [mw]

if RUN_XGB:
    if "qx_model_xgb_mon" in df_final:
        _, mw = mae_cols(df_final, "qx_model_xgb_mon"); mae_bars_labels += ["XGB"]; mae_bars_values += [mw]
    if "qx_model_xgb_wh" in df_final:
        _, mw = mae_cols(df_final, "qx_model_xgb_wh"); mae_bars_labels += ["XGB (WH)"]; mae_bars_values += [mw]

_, mw_tabla = mae_cols(df_final, "qx_tabla_lx")
mae_bars_labels += ["Tabla"]; mae_bars_values += [mw_tabla]

plt.figure(figsize=(7,5))
plt.bar(np.arange(len(mae_bars_labels)), mae_bars_values)
plt.xticks(np.arange(len(mae_bars_labels)), mae_bars_labels)
plt.ylabel("MAE ponderado (vs qₓ Real)")
plt.title("Comparación de MAE ponderado")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mae_weighted_comparison.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,5))
if RUN_NN and "abs_error_nn_wh" in df_final:
    plt.plot(df_final["AGE_BIN"], df_final["abs_error_nn_wh"], marker="o", label="Abs err NN (WH)")
if RUN_XGB and "abs_error_xgb_wh" in df_final:
    plt.plot(df_final["AGE_BIN"], df_final["abs_error_xgb_wh"], marker="s", label="Abs err XGB (WH)")
plt.xticks(rotation=45)
plt.ylabel("|qx_model - qx_real|")
plt.title("Error absoluto por tramo (Whittaker)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "abs_error_wh_by_bin.png"), dpi=300)
plt.close()


print("Saved MAE weighted plot:", "mae_weighted_comparison.png")
print("Listo. Artefactos en:", OUTPUT_DIR)
