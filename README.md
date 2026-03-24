# Credit Default Prediction

Pipeline ML bout-en-bout pour prédire le risque de défaut de crédit.
Dataset : [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
(150 000 clients, 11 variables financières).

## Problématique
Identifier les clients à risque de défaut sur les 2 prochaines années.
Contrainte principale : dataset fortement déséquilibré (6.7% de défauts).

## Pipeline
```
Données brutes → EDA → Preprocessing → Modélisation → SHAP
```

## Résultats

| Modèle | ROC-AUC (CV) | Recall (test) |
|--------|-------------|---------------|
| LogisticRegression | 0.8554 ± 0.0042 | 0.74 |
| RandomForest | 0.8340 ± 0.0019 | 0.15 |
| XGBoost | 0.8644 ± 0.0035 | 0.77 |

Meilleur modèle : **XGBoost** · ROC-AUC = 0.8695 · Recall = 0.785

## Choix techniques
- Métrique : ROC-AUC + Recall (accuracy inadaptée sur données déséquilibrées)
- Preprocessing : Pipeline sklearn — winsorisation → imputation médiane → RobustScaler
- Déséquilibre : `scale_pos_weight` XGBoost + seuil 0.50
- Interprétabilité : SHAP TreeExplainer

## Insights SHAP
Top 3 features les plus prédictives du défaut :
1. `RevolvingUtilizationOfUnsecuredLines` — taux d'utilisation du crédit (0.840)
2. `NumberOfTime30-59DaysPastDueNotWorse` — incidents de paiement passés (0.379)
3. `NumberOfTimes90DaysLate` — retards graves de paiement (0.319)

## Structure
```
credit-default-prediction/
├── notebooks/
│   └── credit-scoring-pipeline.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

## Installation
```bash
git clone https://github.com/kamagatebakagnan/-credit-default-prediction.git
cd -credit-default-prediction
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
jupyter notebook notebooks/credit-scoring-pipeline.ipynb
```
