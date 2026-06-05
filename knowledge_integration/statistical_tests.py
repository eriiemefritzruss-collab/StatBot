from .knw import knw


class Statistical_Test_Assistant(knw):
    def __init__(self):
        super().__init__()
        self.name = "statistical_test_assistant"
        self.description = (
            "Reusable functions for AI + statistics workflows: data overview, "
            "normality checks, two-group comparison, multi-group comparison, "
            "correlation analysis, and categorical association tests. Use this "
            "when the user asks for hypothesis testing, p-values, confidence "
            "intervals, group comparison, correlation, or automatic statistical "
            "method selection."
        )
        self.core_function = "core"
        self.runnable_function = "runnable"
        self.mode = "core"

    def core(self):
        return """
        # Example usage after a DataFrame named data is loaded:
        overview = stat_data_overview(data)
        print(overview["shape"])
        print(overview["missing_values"])

        # Two-group numerical comparison:
        # result = compare_two_groups(data, value_col="score", group_col="group")
        # print(result)

        # Multi-group numerical comparison:
        # result = compare_multiple_groups(data, value_col="score", group_col="treatment")
        # print(result)

        # Categorical association:
        # result = categorical_association(data, row_col="gender", col_col="passed")
        # print(result)

        # Correlation:
        # result = correlation_test(data, x_col="age", y_col="score", method="auto")
        # print(result)
        """

    def runnable(self):
        return r"""
        import numpy as np
        import pandas as pd
        from scipy import stats


        def stat_data_overview(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isna().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "numeric_summary": df[numeric_cols].describe().T if numeric_cols else pd.DataFrame(),
            }


        def _cohens_d(x, y):
            x = pd.Series(x).dropna().astype(float)
            y = pd.Series(y).dropna().astype(float)
            nx, ny = len(x), len(y)
            if nx < 2 or ny < 2:
                return np.nan
            pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
            return np.nan if pooled == 0 else (x.mean() - y.mean()) / pooled


        def normality_by_group(df, value_col, group_col=None, alpha=0.05):
            rows = []
            if group_col is None:
                groups = [("all", df[value_col])]
            else:
                groups = list(df.groupby(group_col, dropna=True)[value_col])
            for group, values in groups:
                values = pd.Series(values).dropna().astype(float)
                if len(values) < 3:
                    rows.append({"group": group, "n": len(values), "test": "not_enough_data", "p_value": np.nan, "normal": False})
                elif len(values) <= 5000:
                    stat, p = stats.shapiro(values)
                    rows.append({"group": group, "n": len(values), "test": "shapiro", "statistic": stat, "p_value": p, "normal": p >= alpha})
                else:
                    stat, p = stats.normaltest(values)
                    rows.append({"group": group, "n": len(values), "test": "dagostino", "statistic": stat, "p_value": p, "normal": p >= alpha})
            return pd.DataFrame(rows)


        def compare_two_groups(df, value_col, group_col, paired=False, alpha=0.05):
            clean = df[[value_col, group_col]].dropna()
            levels = clean[group_col].unique().tolist()
            if len(levels) != 2:
                raise ValueError(f"Expected exactly 2 groups, got {len(levels)} groups: {levels}")

            x = clean.loc[clean[group_col] == levels[0], value_col].astype(float)
            y = clean.loc[clean[group_col] == levels[1], value_col].astype(float)
            normal = normality_by_group(clean, value_col, group_col, alpha)

            if paired:
                n = min(len(x), len(y))
                x2, y2 = x.iloc[:n], y.iloc[:n]
                diff = x2.reset_index(drop=True) - y2.reset_index(drop=True)
                if len(diff) >= 3 and stats.shapiro(diff).pvalue >= alpha:
                    stat, p = stats.ttest_rel(x2, y2)
                    test = "paired_t_test"
                else:
                    stat, p = stats.wilcoxon(x2, y2)
                    test = "wilcoxon_signed_rank"
            else:
                levene_p = stats.levene(x, y).pvalue if len(x) > 1 and len(y) > 1 else np.nan
                if normal["normal"].all() and (np.isnan(levene_p) or levene_p >= alpha):
                    stat, p = stats.ttest_ind(x, y, equal_var=True)
                    test = "independent_t_test"
                elif normal["normal"].all():
                    stat, p = stats.ttest_ind(x, y, equal_var=False)
                    test = "welch_t_test"
                else:
                    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                    test = "mann_whitney_u"

            return {
                "test": test,
                "groups": levels,
                "n": {str(levels[0]): int(len(x)), str(levels[1]): int(len(y))},
                "means": {str(levels[0]): float(x.mean()), str(levels[1]): float(y.mean())},
                "statistic": float(stat),
                "p_value": float(p),
                "cohens_d": float(_cohens_d(x, y)),
                "normality": normal,
            }


        def compare_multiple_groups(df, value_col, group_col, alpha=0.05):
            clean = df[[value_col, group_col]].dropna()
            groups = [g[value_col].astype(float) for _, g in clean.groupby(group_col)]
            labels = [str(label) for label, _ in clean.groupby(group_col)]
            if len(groups) < 3:
                raise ValueError("Use compare_two_groups for fewer than 3 groups.")
            normal = normality_by_group(clean, value_col, group_col, alpha)
            levene_stat, levene_p = stats.levene(*groups)
            if normal["normal"].all() and levene_p >= alpha:
                stat, p = stats.f_oneway(*groups)
                test = "one_way_anova"
            else:
                stat, p = stats.kruskal(*groups)
                test = "kruskal_wallis"
            return {
                "test": test,
                "groups": labels,
                "group_sizes": {label: int(len(group)) for label, group in zip(labels, groups)},
                "statistic": float(stat),
                "p_value": float(p),
                "levene_p_value": float(levene_p),
                "normality": normal,
            }


        def categorical_association(df, row_col, col_col):
            table = pd.crosstab(df[row_col], df[col_col])
            if table.shape == (2, 2) and (table.values < 5).any():
                odds_ratio, p = stats.fisher_exact(table)
                return {"test": "fisher_exact", "table": table, "odds_ratio": float(odds_ratio), "p_value": float(p)}
            chi2, p, dof, expected = stats.chi2_contingency(table)
            n = table.values.sum()
            phi2 = chi2 / n
            r, k = table.shape
            cramers_v = np.sqrt(phi2 / max(1, min(k - 1, r - 1)))
            return {
                "test": "chi_square",
                "table": table,
                "chi2": float(chi2),
                "p_value": float(p),
                "dof": int(dof),
                "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
                "cramers_v": float(cramers_v),
            }


        def correlation_test(df, x_col, y_col, method="auto"):
            clean = df[[x_col, y_col]].dropna().astype(float)
            if method == "auto":
                nx = len(clean)
                if nx >= 3 and nx <= 5000:
                    normal_x = stats.shapiro(clean[x_col]).pvalue >= 0.05
                    normal_y = stats.shapiro(clean[y_col]).pvalue >= 0.05
                else:
                    normal_x = normal_y = False
                method = "pearson" if normal_x and normal_y else "spearman"
            if method == "pearson":
                stat, p = stats.pearsonr(clean[x_col], clean[y_col])
            elif method == "spearman":
                stat, p = stats.spearmanr(clean[x_col], clean[y_col])
            else:
                stat, p = stats.kendalltau(clean[x_col], clean[y_col])
            return {"method": method, "n": int(len(clean)), "correlation": float(stat), "p_value": float(p)}
        """
