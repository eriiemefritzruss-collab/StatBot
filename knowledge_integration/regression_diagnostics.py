from .knw import knw


class Regression_Diagnostics(knw):
    def __init__(self):
        super().__init__()
        self.name = "regression_diagnostics"
        self.description = (
            "Reusable OLS regression diagnostics for AI + statistics workflows: "
            "fit linear regression with statsmodels, summarize coefficients, "
            "confidence intervals, residual normality, heteroskedasticity, "
            "Durbin-Watson, VIF multicollinearity, influential observations, "
            "and diagnostic plots. Use this when the user asks for regression, "
            "coefficient interpretation, model assumptions, residual analysis, "
            "or robust statistical modeling."
        )
        self.core_function = "core"
        self.runnable_function = "runnable"
        self.mode = "core"

    def core(self):
        return """
        # Example usage after a DataFrame named data is loaded:
        # result = fit_ols_with_diagnostics(data, formula="y ~ x1 + x2")
        # print(result["coefficient_table"])
        # print(result["diagnostics"])
        # result["model"].summary()

        # To generate diagnostic plots:
        # plot_ols_diagnostics(result["model"])
        """

    def runnable(self):
        return r"""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        from scipy import stats
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.stattools import durbin_watson


        def _coefficient_table(model):
            conf = model.conf_int()
            term_names = getattr(model.params, "index", None)
            if term_names is None:
                term_names = model.model.exog_names
            table = pd.DataFrame({
                "term": list(term_names),
                "coef": np.asarray(model.params),
                "std_err": np.asarray(model.bse),
                "t": np.asarray(model.tvalues),
                "p_value": np.asarray(model.pvalues),
                "ci_low": np.asarray(conf)[:, 0],
                "ci_high": np.asarray(conf)[:, 1],
            })
            return table


        def _vif_table(model):
            exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
            if "Intercept" in exog.columns:
                vif_exog = exog.drop(columns=["Intercept"])
            elif "const" in exog.columns:
                vif_exog = exog.drop(columns=["const"])
            else:
                vif_exog = exog
            numeric = vif_exog.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                return pd.DataFrame(columns=["variable", "vif"])
            rows = []
            for i, col in enumerate(numeric.columns):
                try:
                    vif = variance_inflation_factor(numeric.values, i)
                except Exception:
                    vif = np.nan
                rows.append({"variable": col, "vif": vif})
            return pd.DataFrame(rows)


        def fit_ols_with_diagnostics(df, formula, robust_cov=None):
            clean = df.dropna()
            model = smf.ols(formula=formula, data=clean).fit()
            if robust_cov:
                model = model.get_robustcov_results(cov_type=robust_cov)

            resid = pd.Series(model.resid).dropna()
            shapiro_p = stats.shapiro(resid).pvalue if 3 <= len(resid) <= 5000 else np.nan
            bp = het_breuschpagan(model.resid, model.model.exog)
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]

            diagnostics = {
                "n_obs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "residual_shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
                "breusch_pagan_lm_p": float(bp[1]),
                "breusch_pagan_f_p": float(bp[3]),
                "durbin_watson": float(durbin_watson(model.resid)),
                "max_cooks_distance": float(np.nanmax(cooks_d)),
                "n_high_influence_cooks_gt_4_over_n": int(np.sum(cooks_d > 4 / max(1, len(cooks_d)))),
            }

            return {
                "model": model,
                "coefficient_table": _coefficient_table(model),
                "diagnostics": diagnostics,
                "vif": _vif_table(model),
            }


        def plot_ols_diagnostics(model):
            fitted = model.fittedvalues
            resid = model.resid
            standardized_resid = model.get_influence().resid_studentized_internal

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            sns.scatterplot(x=fitted, y=resid, ax=axes[0, 0])
            axes[0, 0].axhline(0, color="red", linestyle="--")
            axes[0, 0].set_title("Residuals vs Fitted")
            axes[0, 0].set_xlabel("Fitted values")
            axes[0, 0].set_ylabel("Residuals")

            sm.qqplot(resid, line="45", ax=axes[0, 1])
            axes[0, 1].set_title("Normal Q-Q")

            sns.scatterplot(x=fitted, y=np.sqrt(np.abs(standardized_resid)), ax=axes[1, 0])
            axes[1, 0].set_title("Scale-Location")
            axes[1, 0].set_xlabel("Fitted values")
            axes[1, 0].set_ylabel("sqrt(abs(Standardized residuals))")

            cooks_d = model.get_influence().cooks_distance[0]
            axes[1, 1].stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
            axes[1, 1].axhline(4 / max(1, len(cooks_d)), color="red", linestyle="--")
            axes[1, 1].set_title("Cook's Distance")
            axes[1, 1].set_xlabel("Observation index")
            axes[1, 1].set_ylabel("Cook's D")

            plt.tight_layout()
            plt.show()
            return fig
        """
