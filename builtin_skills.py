import re
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class AnalysisContext:
    user_message: str
    dataframe: pd.DataFrame
    message: str
    columns: list
    numeric_cols: list
    categorical_cols: list
    matched_cols: list
    matched_numeric_cols: list
    matched_categorical_cols: list
    time_cols: list


@dataclass(frozen=True)
class AnalysisToolSpec:
    name: str
    category: str
    description: str
    matcher: Callable[[AnalysisContext], bool]
    builder: Callable[[AnalysisContext], Optional[dict]]


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _mentioned_columns(user_message, columns):
    lowered = user_message.lower()
    matched = [col for col in columns if str(col).lower() in lowered]
    return matched


def _numeric_columns(dataframe):
    return list(dataframe.select_dtypes(include=["number"]).columns)


def _categorical_columns(dataframe):
    return list(dataframe.select_dtypes(exclude=["number"]).columns)


def _time_columns(dataframe):
    candidates = []
    for col in dataframe.columns:
        col_name = str(col).lower()
        if any(token in col_name for token in ["date", "time", "month", "day", "year"]):
            candidates.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
            candidates.append(col)
    return candidates


def _extract_first_number(text):
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _pick_numeric(ctx, exclude=None):
    exclude = set(exclude or [])
    for col in ctx.matched_numeric_cols:
        if col not in exclude:
            return col
    for col in ctx.numeric_cols:
        if col not in exclude:
            return col
    return None


def _pick_two_numeric(ctx):
    seen = []
    for col in ctx.matched_numeric_cols + ctx.numeric_cols:
        if col not in seen:
            seen.append(col)
    return seen[:2]


def _pick_categorical(ctx, exclude=None):
    exclude = set(exclude or [])
    for col in ctx.matched_categorical_cols:
        if col not in exclude:
            return col
    for col in ctx.categorical_cols:
        if col not in exclude:
            return col
    return None


def _pick_two_categorical(ctx):
    seen = []
    for col in ctx.matched_categorical_cols + ctx.categorical_cols:
        if col not in seen:
            seen.append(col)
    return seen[:2]


def _pick_binary_categorical(ctx, exclude=None):
    exclude = set(exclude or [])
    for col in ctx.matched_categorical_cols + ctx.categorical_cols:
        if col in exclude:
            continue
        if ctx.dataframe[col].dropna().nunique() == 2:
            return col
    return None


def _pick_time_col(ctx):
    if ctx.time_cols:
        for col in ctx.matched_cols:
            if col in ctx.time_cols:
                return col
        return ctx.time_cols[0]
    return None


def _build_context(user_message, dataframe):
    columns = list(dataframe.columns)
    numeric_cols = _numeric_columns(dataframe)
    categorical_cols = _categorical_columns(dataframe)
    matched_cols = _mentioned_columns(user_message, columns)
    return AnalysisContext(
        user_message=user_message,
        dataframe=dataframe,
        message=_normalize(user_message),
        columns=columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        matched_cols=matched_cols,
        matched_numeric_cols=[col for col in matched_cols if col in numeric_cols],
        matched_categorical_cols=[col for col in matched_cols if col in categorical_cols],
        time_cols=_time_columns(dataframe),
    )


def _wrap_tool(spec, rationale, assumptions, code):
    return {
        "name": spec.name,
        "category": spec.category,
        "description": spec.description,
        "rationale": rationale,
        "assumptions": assumptions,
        "code": code,
    }


def _build_dataset_overview(ctx, spec):
    return _wrap_tool(
        spec,
        "The request asks for a dataset overview or data quality summary.",
        "This descriptive overview does not rely on modeling assumptions.",
        (
            "overview = pd.DataFrame({\n"
            "    'column': data.columns,\n"
            "    'dtype': data.dtypes.astype(str).values,\n"
            "    'missing_values': data.isnull().sum().values,\n"
            "    'missing_rate': (data.isnull().mean() * 100).round(2).values,\n"
            "    'n_unique': data.nunique(dropna=True).values,\n"
            "})\n"
            "print(f'Rows: {data.shape[0]}, Columns: {data.shape[1]}')\n"
            "print(f'Duplicate rows: {int(data.duplicated().sum())}')\n"
            "print('\\nColumn overview:')\n"
            "overview\n"
            "print('\\nNumeric summary:')\n"
            "display(data.describe(include='number').T if not data.select_dtypes(include=['number']).empty else pd.DataFrame())\n"
            "print('\\nCategorical summary:')\n"
            "display(data.describe(include='object').T if not data.select_dtypes(exclude=['number']).empty else pd.DataFrame())\n"
            "print('\\nPreview:')\n"
            "data.head()"
        ),
    )


def _build_descriptive_statistics(ctx, spec):
    numeric_cols = ctx.matched_numeric_cols or ctx.numeric_cols[:8]
    categorical_cols = ctx.matched_categorical_cols or ctx.categorical_cols[:5]
    return _wrap_tool(
        spec,
        "The request is for descriptive or summary statistics.",
        "This is descriptive analysis, so no inferential assumptions are required.",
        (
            f"numeric_cols = {repr(numeric_cols)}\n"
            f"categorical_cols = {repr(categorical_cols)}\n"
            "if numeric_cols:\n"
            "    numeric_summary = data[numeric_cols].describe().T\n"
            "    numeric_summary['skew'] = data[numeric_cols].skew(numeric_only=True)\n"
            "    numeric_summary['kurtosis'] = data[numeric_cols].kurtosis(numeric_only=True)\n"
            "    print('Numeric summary statistics:')\n"
            "    display(numeric_summary)\n"
            "if categorical_cols:\n"
            "    for col in categorical_cols:\n"
            "        print(f'\\nTop frequencies for {col}:')\n"
            "        display(data[col].value_counts(dropna=False).head(10).rename('count').to_frame())"
        ),
    )


def _build_missingness_profile(ctx, spec):
    return _wrap_tool(
        spec,
        "The request asks about missing values, completeness, or data quality issues.",
        "This is a descriptive audit of missingness patterns rather than a formal missing-data mechanism test.",
        (
            "missing_summary = pd.DataFrame({\n"
            "    'column': data.columns,\n"
            "    'missing_count': data.isnull().sum().values,\n"
            "    'missing_rate_pct': (data.isnull().mean() * 100).round(2).values,\n"
            "}).sort_values(['missing_count', 'column'], ascending=[False, True])\n"
            "display(missing_summary)\n"
            "plt.figure(figsize=(10, max(3, len(data.columns) * 0.3)))\n"
            "sns.heatmap(data.isnull().T, cbar=False, cmap='viridis')\n"
            "plt.title('Missingness Pattern by Column and Row')\n"
            "plt.xlabel('Row Index')\n"
            "plt.ylabel('Column')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_outlier_screening(ctx, spec):
    numeric_cols = ctx.matched_numeric_cols or ctx.numeric_cols[:6]
    if not numeric_cols:
        return None
    return _wrap_tool(
        spec,
        "The request mentions outliers, unusual values, or distribution screening.",
        "The IQR and z-score rules are heuristic screening tools and should be combined with domain judgment.",
        (
            f"numeric_cols = {repr(numeric_cols)}\n"
            "rows = []\n"
            "for col in numeric_cols:\n"
            "    series = data[col].dropna().astype(float)\n"
            "    if len(series) < 4:\n"
            "        continue\n"
            "    q1 = series.quantile(0.25)\n"
            "    q3 = series.quantile(0.75)\n"
            "    iqr = q3 - q1\n"
            "    lower = q1 - 1.5 * iqr\n"
            "    upper = q3 + 1.5 * iqr\n"
            "    iqr_outliers = int(((series < lower) | (series > upper)).sum())\n"
            "    z = np.abs(stats.zscore(series, nan_policy='omit')) if len(series) > 1 else np.array([])\n"
            "    z_outliers = int((z > 3).sum()) if len(z) else 0\n"
            "    rows.append({\n"
            "        'variable': col,\n"
            "        'n': len(series),\n"
            "        'iqr_outliers': iqr_outliers,\n"
            "        'zscore_outliers_gt3': z_outliers,\n"
            "        'lower_fence': lower,\n"
            "        'upper_fence': upper,\n"
            "    })\n"
            "summary = pd.DataFrame(rows).sort_values(['iqr_outliers', 'zscore_outliers_gt3'], ascending=False)\n"
            "display(summary)\n"
            "melted = data[numeric_cols].melt(var_name='variable', value_name='value').dropna()\n"
            "plt.figure(figsize=(10, 4))\n"
            "sns.boxplot(data=melted, x='variable', y='value')\n"
            "plt.xticks(rotation=45)\n"
            "plt.title('Outlier Screening Across Numeric Variables')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_normality_check(ctx, spec):
    target_col = _pick_numeric(ctx)
    if target_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request mentions normality or distributional assumptions.",
        "Shapiro-Wilk is most suitable for smaller samples; visual checks remain important for larger samples.",
        (
            f"target_col = {target_col!r}\n"
            "series = data[target_col].dropna().astype(float)\n"
            "sample = series.sample(min(len(series), 5000), random_state=42) if len(series) > 5000 else series\n"
            "shapiro_stat, shapiro_p = stats.shapiro(sample)\n"
            "normaltest_stat, normaltest_p = stats.normaltest(series) if len(series) >= 8 else (np.nan, np.nan)\n"
            "summary = pd.DataFrame([\n"
            "    {'test': 'Shapiro-Wilk', 'statistic': shapiro_stat, 'p_value': shapiro_p},\n"
            "    {'test': 'D\\'Agostino K^2', 'statistic': normaltest_stat, 'p_value': normaltest_p},\n"
            "])\n"
            "display(summary)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "sns.histplot(series, kde=True, ax=axes[0])\n"
            "axes[0].set_title(f'Distribution of {target_col}')\n"
            "stats.probplot(series, dist='norm', plot=axes[1])\n"
            "axes[1].set_title('Q-Q Plot')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_one_sample_t_test(ctx, spec):
    target_col = _pick_numeric(ctx)
    reference_value = _extract_first_number(ctx.user_message)
    if target_col is None or reference_value is None:
        return None
    return _wrap_tool(
        spec,
        "The request matches a one-sample hypothesis test with a reference value.",
        "The observations should be independent, and the sample mean should be reasonably modeled by a normal distribution or a large-sample approximation.",
        (
            f"target_col = {target_col!r}\n"
            f"reference_value = {reference_value!r}\n"
            "sample = data[target_col].dropna().astype(float)\n"
            "test_stat, p_value = stats.ttest_1samp(sample, popmean=reference_value)\n"
            "mean_diff = sample.mean() - reference_value\n"
            "se = sample.std(ddof=1) / np.sqrt(len(sample))\n"
            "t_crit = stats.t.ppf(0.975, df=len(sample) - 1)\n"
            "ci_low = mean_diff - t_crit * se\n"
            "ci_high = mean_diff + t_crit * se\n"
            "cohens_d = mean_diff / sample.std(ddof=1)\n"
            "result = pd.DataFrame([{\n"
            "    'variable': target_col,\n"
            "    'reference_value': reference_value,\n"
            "    'sample_mean': sample.mean(),\n"
            "    't_statistic': test_stat,\n"
            "    'p_value': p_value,\n"
            "    'cohens_d': cohens_d,\n"
            "    'ci_low_mean_diff': ci_low,\n"
            "    'ci_high_mean_diff': ci_high,\n"
            "}])\n"
            "print('H0: the population mean equals the reference value.')\n"
            "print('H1: the population mean differs from the reference value.')\n"
            "display(result)"
        ),
    )


def _build_two_sample_t_test(ctx, spec):
    outcome_col = _pick_numeric(ctx)
    group_col = _pick_categorical(ctx)
    if outcome_col is None or group_col is None:
        return None
    if ctx.dataframe[group_col].dropna().nunique() < 2:
        return None
    return _wrap_tool(
        spec,
        "The request looks like a two-group mean comparison.",
        "The groups should be independent, and the outcome should be approximately normal within each group or supported by sample size.",
        (
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "group_levels = sorted(temp[group_col].unique().tolist())\n"
            "selected_levels = group_levels[:2]\n"
            "group_a = temp.loc[temp[group_col] == selected_levels[0], outcome_col].astype(float)\n"
            "group_b = temp.loc[temp[group_col] == selected_levels[1], outcome_col].astype(float)\n"
            "levene_stat, levene_p = stats.levene(group_a, group_b, center='median')\n"
            "equal_var = bool(levene_p >= 0.05)\n"
            "test_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var, nan_policy='omit')\n"
            "mean_diff = group_a.mean() - group_b.mean()\n"
            "pooled_sd = np.sqrt(((len(group_a)-1) * group_a.var(ddof=1) + (len(group_b)-1) * group_b.var(ddof=1)) / (len(group_a) + len(group_b) - 2))\n"
            "cohens_d = mean_diff / pooled_sd if pooled_sd else np.nan\n"
            "se = np.sqrt(group_a.var(ddof=1) / len(group_a) + group_b.var(ddof=1) / len(group_b))\n"
            "df = len(group_a) + len(group_b) - 2\n"
            "t_crit = stats.t.ppf(0.975, df=df)\n"
            "ci_low = mean_diff - t_crit * se\n"
            "ci_high = mean_diff + t_crit * se\n"
            "summary = temp.groupby(group_col)[outcome_col].agg(['count', 'mean', 'std']).reset_index()\n"
            "result = pd.DataFrame([{\n"
            "    'group_1': selected_levels[0],\n"
            "    'group_2': selected_levels[1],\n"
            "    'mean_difference': mean_diff,\n"
            "    't_statistic': test_stat,\n"
            "    'p_value': p_value,\n"
            "    'equal_variance_assumed': equal_var,\n"
            "    'cohens_d': cohens_d,\n"
            "    'ci_low_mean_diff': ci_low,\n"
            "    'ci_high_mean_diff': ci_high,\n"
            "}])\n"
            "print('H0: the two group means are equal.')\n"
            "print('H1: the two group means differ.')\n"
            "display(summary)\n"
            "display(result)"
        ),
    )


def _build_variance_homogeneity(ctx, spec):
    outcome_col = _pick_numeric(ctx)
    group_col = _pick_categorical(ctx)
    if outcome_col is None or group_col is None:
        return None
    if ctx.dataframe[group_col].dropna().nunique() < 2:
        return None
    return _wrap_tool(
        spec,
        "The request is about equal variances, homoscedasticity, or variance assumptions across groups.",
        "Levene is more robust than Bartlett when normality is uncertain; these tests only address group variance equality.",
        (
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "groups = [grp[outcome_col].astype(float).values for _, grp in temp.groupby(group_col)]\n"
            "levene_stat, levene_p = stats.levene(*groups, center='median')\n"
            "bartlett_stat, bartlett_p = stats.bartlett(*groups) if all(len(g) >= 2 for g in groups) else (np.nan, np.nan)\n"
            "group_summary = temp.groupby(group_col)[outcome_col].agg(['count', 'mean', 'std', 'var']).reset_index()\n"
            "result = pd.DataFrame([\n"
            "    {'test': 'Levene', 'statistic': levene_stat, 'p_value': levene_p},\n"
            "    {'test': 'Bartlett', 'statistic': bartlett_stat, 'p_value': bartlett_p},\n"
            "])\n"
            "display(group_summary)\n"
            "display(result)\n"
            "plt.figure(figsize=(9, 4))\n"
            "sns.boxplot(data=temp, x=group_col, y=outcome_col)\n"
            "plt.xticks(rotation=45)\n"
            "plt.title(f'Variance Comparison for {outcome_col} by {group_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_paired_t_test(ctx, spec):
    numeric_cols = _pick_two_numeric(ctx)
    if len(numeric_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request suggests a paired comparison such as before-vs-after.",
        "The paired differences should be approximately normal, and each row should represent a matched pair.",
        (
            f"col_a = {numeric_cols[0]!r}\n"
            f"col_b = {numeric_cols[1]!r}\n"
            "temp = data[[col_a, col_b]].dropna().astype(float)\n"
            "test_stat, p_value = stats.ttest_rel(temp[col_a], temp[col_b])\n"
            "diff = temp[col_a] - temp[col_b]\n"
            "mean_diff = diff.mean()\n"
            "se = diff.std(ddof=1) / np.sqrt(len(diff))\n"
            "t_crit = stats.t.ppf(0.975, df=len(diff) - 1)\n"
            "ci_low = mean_diff - t_crit * se\n"
            "ci_high = mean_diff + t_crit * se\n"
            "cohens_d_z = mean_diff / diff.std(ddof=1)\n"
            "result = pd.DataFrame([{\n"
            "    'pair_a': col_a,\n"
            "    'pair_b': col_b,\n"
            "    'mean_difference': mean_diff,\n"
            "    't_statistic': test_stat,\n"
            "    'p_value': p_value,\n"
            "    'cohens_dz': cohens_d_z,\n"
            "    'ci_low_mean_diff': ci_low,\n"
            "    'ci_high_mean_diff': ci_high,\n"
            "}])\n"
            "print('H0: the mean paired difference is zero.')\n"
            "print('H1: the mean paired difference is not zero.')\n"
            "display(result)"
        ),
    )


def _build_wilcoxon_signed_rank(ctx, spec):
    numeric_cols = _pick_two_numeric(ctx)
    if len(numeric_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request suggests a nonparametric paired comparison.",
        "Wilcoxon signed-rank assumes matched pairs and interprets differences in paired ranks rather than means.",
        (
            f"col_a = {numeric_cols[0]!r}\n"
            f"col_b = {numeric_cols[1]!r}\n"
            "temp = data[[col_a, col_b]].dropna().astype(float)\n"
            "statistic, p_value = stats.wilcoxon(temp[col_a], temp[col_b], zero_method='wilcox', alternative='two-sided')\n"
            "diff = temp[col_a] - temp[col_b]\n"
            "result = pd.DataFrame([{\n"
            "    'pair_a': col_a,\n"
            "    'pair_b': col_b,\n"
            "    'wilcoxon_statistic': statistic,\n"
            "    'p_value': p_value,\n"
            "    'median_difference': diff.median(),\n"
            "}])\n"
            "print('H0: the paired distribution of differences is centered at zero.')\n"
            "print('H1: the paired distribution of differences is not centered at zero.')\n"
            "display(result)"
        ),
    )


def _build_mann_whitney(ctx, spec):
    outcome_col = _pick_numeric(ctx)
    group_col = _pick_categorical(ctx)
    if outcome_col is None or group_col is None:
        return None
    if ctx.dataframe[group_col].dropna().nunique() < 2:
        return None
    return _wrap_tool(
        spec,
        "The request mentions a nonparametric two-group comparison.",
        "This test assumes independent observations and similar distribution shapes when interpreted as a location shift.",
        (
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "group_levels = sorted(temp[group_col].unique().tolist())[:2]\n"
            "group_a = temp.loc[temp[group_col] == group_levels[0], outcome_col].astype(float)\n"
            "group_b = temp.loc[temp[group_col] == group_levels[1], outcome_col].astype(float)\n"
            "u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')\n"
            "result = pd.DataFrame([{\n"
            "    'group_1': group_levels[0],\n"
            "    'group_2': group_levels[1],\n"
            "    'u_statistic': u_stat,\n"
            "    'p_value': p_value,\n"
            "}])\n"
            "display(temp.groupby(group_col)[outcome_col].agg(['count', 'median', 'mean']).reset_index())\n"
            "print('H0: the two groups come from the same distribution.')\n"
            "print('H1: the two groups differ in distribution/location.')\n"
            "display(result)"
        ),
    )


def _build_anova(ctx, spec):
    outcome_col = _pick_numeric(ctx)
    group_col = _pick_categorical(ctx)
    if outcome_col is None or group_col is None:
        return None
    if ctx.dataframe[group_col].dropna().nunique() < 3:
        return None
    return _wrap_tool(
        spec,
        "The request matches a multi-group mean comparison.",
        "ANOVA assumes independent observations, approximate normality within groups, and reasonably similar variances.",
        (
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "groups = [grp[outcome_col].astype(float).values for _, grp in temp.groupby(group_col)]\n"
            "f_stat, p_value = stats.f_oneway(*groups)\n"
            "group_means = temp.groupby(group_col)[outcome_col].mean()\n"
            "grand_mean = temp[outcome_col].mean()\n"
            "ss_between = sum(temp.groupby(group_col).size()[name] * (mean - grand_mean) ** 2 for name, mean in group_means.items())\n"
            "ss_total = ((temp[outcome_col] - grand_mean) ** 2).sum()\n"
            "eta_squared = ss_between / ss_total if ss_total else np.nan\n"
            "summary = temp.groupby(group_col)[outcome_col].agg(['count', 'mean', 'std']).reset_index()\n"
            "result = pd.DataFrame([{'f_statistic': f_stat, 'p_value': p_value, 'eta_squared': eta_squared}])\n"
            "print('H0: all group means are equal.')\n"
            "print('H1: at least one group mean differs.')\n"
            "display(summary)\n"
            "display(result)"
        ),
    )


def _build_kruskal_wallis(ctx, spec):
    outcome_col = _pick_numeric(ctx)
    group_col = _pick_categorical(ctx)
    if outcome_col is None or group_col is None:
        return None
    if ctx.dataframe[group_col].dropna().nunique() < 3:
        return None
    return _wrap_tool(
        spec,
        "The request points to a nonparametric comparison across multiple groups.",
        "Kruskal-Wallis assumes independent observations and is most naturally interpreted as a group location/distribution comparison.",
        (
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "groups = [grp[outcome_col].astype(float).values for _, grp in temp.groupby(group_col)]\n"
            "h_stat, p_value = stats.kruskal(*groups)\n"
            "summary = temp.groupby(group_col)[outcome_col].agg(['count', 'median', 'mean', 'std']).reset_index()\n"
            "result = pd.DataFrame([{'h_statistic': h_stat, 'p_value': p_value}])\n"
            "print('H0: all groups come from the same distribution.')\n"
            "print('H1: at least one group differs in distribution/location.')\n"
            "display(summary)\n"
            "display(result)"
        ),
    )


def _build_chi_square(ctx, spec):
    categorical_cols = _pick_two_categorical(ctx)
    if len(categorical_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request is about association between categorical variables.",
        "The chi-square approximation is most reliable when expected cell counts are not too small.",
        (
            f"row_col = {categorical_cols[0]!r}\n"
            f"col_col = {categorical_cols[1]!r}\n"
            "temp = data[[row_col, col_col]].dropna().copy()\n"
            "temp[row_col] = temp[row_col].astype(str)\n"
            "temp[col_col] = temp[col_col].astype(str)\n"
            "contingency = pd.crosstab(temp[row_col], temp[col_col])\n"
            "chi2, p_value, dof, expected = stats.chi2_contingency(contingency)\n"
            "n = contingency.to_numpy().sum()\n"
            "phi2 = chi2 / n if n else np.nan\n"
            "r, c = contingency.shape\n"
            "cramers_v = np.sqrt(phi2 / max(min(r - 1, c - 1), 1)) if n else np.nan\n"
            "expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)\n"
            "result = pd.DataFrame([{\n"
            "    'chi_square': chi2,\n"
            "    'degrees_of_freedom': dof,\n"
            "    'p_value': p_value,\n"
            "    'cramers_v': cramers_v,\n"
            "}])\n"
            "print('H0: the two categorical variables are independent.')\n"
            "print('H1: the two categorical variables are associated.')\n"
            "display(contingency)\n"
            "print('\\nExpected frequencies:')\n"
            "display(expected_df)\n"
            "display(result)"
        ),
    )


def _build_fisher_exact(ctx, spec):
    row_col = _pick_binary_categorical(ctx)
    col_col = _pick_binary_categorical(ctx, exclude=[row_col] if row_col else [])
    if row_col is None or col_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is for an exact 2x2 association test between binary categorical variables.",
        "Fisher's exact test is tailored to 2x2 tables and is especially helpful when expected counts are small.",
        (
            f"row_col = {row_col!r}\n"
            f"col_col = {col_col!r}\n"
            "temp = data[[row_col, col_col]].dropna().copy()\n"
            "temp[row_col] = temp[row_col].astype(str)\n"
            "temp[col_col] = temp[col_col].astype(str)\n"
            "table = pd.crosstab(temp[row_col], temp[col_col])\n"
            "if table.shape != (2, 2):\n"
            "    raise ValueError('Fisher exact test requires a 2x2 contingency table.')\n"
            "odds_ratio, p_value = stats.fisher_exact(table)\n"
            "result = pd.DataFrame([{'odds_ratio': odds_ratio, 'p_value': p_value}])\n"
            "print('H0: the two binary variables are independent.')\n"
            "print('H1: the two binary variables are associated.')\n"
            "display(table)\n"
            "display(result)"
        ),
    )


def _build_two_proportion_test(ctx, spec):
    outcome_col = _pick_binary_categorical(ctx)
    group_col = _pick_binary_categorical(ctx, exclude=[outcome_col] if outcome_col else [])
    if outcome_col is None or group_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is a binary proportion comparison across two groups.",
        "This z-test compares proportions under an approximate normal assumption for the group event rates.",
        (
            "from statsmodels.stats.proportion import proportions_ztest\n"
            f"outcome_col = {outcome_col!r}\n"
            f"group_col = {group_col!r}\n"
            "temp = data[[outcome_col, group_col]].dropna().copy()\n"
            "temp[outcome_col] = temp[outcome_col].astype(str)\n"
            "temp[group_col] = temp[group_col].astype(str)\n"
            "group_levels = sorted(temp[group_col].unique().tolist())[:2]\n"
            "outcome_levels = sorted(temp[outcome_col].unique().tolist())[:2]\n"
            "success_level = outcome_levels[-1]\n"
            "counts = []\n"
            "nobs = []\n"
            "rows = []\n"
            "for level in group_levels:\n"
            "    subset = temp.loc[temp[group_col] == level, outcome_col]\n"
            "    success_count = int((subset == success_level).sum())\n"
            "    counts.append(success_count)\n"
            "    nobs.append(int(len(subset)))\n"
            "    rows.append({'group': level, 'successes': success_count, 'n': int(len(subset)), 'proportion': success_count / len(subset) if len(subset) else np.nan})\n"
            "z_stat, p_value = proportions_ztest(counts, nobs)\n"
            "summary = pd.DataFrame(rows)\n"
            "result = pd.DataFrame([{'success_level': success_level, 'z_statistic': z_stat, 'p_value': p_value}])\n"
            "print('H0: the success proportions are equal across the two groups.')\n"
            "print('H1: the success proportions differ across the two groups.')\n"
            "display(summary)\n"
            "display(result)"
        ),
    )


def _build_correlation(ctx, spec):
    cols = ctx.matched_numeric_cols if len(ctx.matched_numeric_cols) >= 2 else ctx.numeric_cols[:8]
    if len(cols) < 2:
        return None
    method = "spearman" if "spearman" in ctx.message else "pearson"
    return _wrap_tool(
        spec,
        f"The request is a {method} correlation analysis over numeric columns.",
        "Pearson targets linear association on numeric scales; Spearman is rank-based and more robust to non-normality and monotonic relationships.",
        (
            f"selected_cols = {repr(cols)}\n"
            f"method = {method!r}\n"
            "corr = data[selected_cols].corr(method=method, numeric_only=True)\n"
            "display(corr)\n"
            "plt.figure(figsize=(8, 6))\n"
            "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')\n"
            "plt.title(f'{method.title()} Correlation Matrix')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_grouped_summary(ctx, spec):
    group_col = _pick_categorical(ctx)
    value_col = _pick_numeric(ctx)
    if group_col is None or value_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is a grouped descriptive comparison.",
        "This grouped summary is descriptive; inferential claims would need a separate test.",
        (
            f"group_col = {group_col!r}\n"
            f"value_col = {value_col!r}\n"
            "summary = data.groupby(group_col)[value_col].agg(['count', 'mean', 'median', 'std', 'sum']).reset_index()\n"
            "display(summary)\n"
            "plt.figure(figsize=(8, 4))\n"
            "sns.barplot(data=summary, x=group_col, y='mean')\n"
            "plt.xticks(rotation=45)\n"
            "plt.title(f'Mean {value_col} by {group_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_categorical_frequency(ctx, spec):
    group_col = _pick_categorical(ctx)
    if group_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request asks for counts, proportions, or categorical composition.",
        "This is a descriptive frequency summary and does not itself imply statistical significance.",
        (
            f"group_col = {group_col!r}\n"
            "counts = data[group_col].astype(str).value_counts(dropna=False).rename_axis(group_col).reset_index(name='count')\n"
            "counts['proportion'] = counts['count'] / counts['count'].sum()\n"
            "display(counts)\n"
            "plt.figure(figsize=(8, 4))\n"
            "sns.countplot(data=data.assign(**{group_col: data[group_col].astype(str)}), x=group_col, order=counts[group_col])\n"
            "plt.xticks(rotation=45)\n"
            "plt.title(f'Frequency Distribution of {group_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_crosstab_summary(ctx, spec):
    categorical_cols = _pick_two_categorical(ctx)
    if len(categorical_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request is a cross-tabulation or segmented categorical summary.",
        "This is descriptive cross-tabulation; association testing would use chi-square or Fisher's exact test.",
        (
            f"row_col = {categorical_cols[0]!r}\n"
            f"col_col = {categorical_cols[1]!r}\n"
            "temp = data[[row_col, col_col]].dropna().copy()\n"
            "temp[row_col] = temp[row_col].astype(str)\n"
            "temp[col_col] = temp[col_col].astype(str)\n"
            "counts = pd.crosstab(temp[row_col], temp[col_col])\n"
            "row_pct = pd.crosstab(temp[row_col], temp[col_col], normalize='index').round(3)\n"
            "col_pct = pd.crosstab(temp[row_col], temp[col_col], normalize='columns').round(3)\n"
            "print('Counts:')\n"
            "display(counts)\n"
            "print('\\nRow percentages:')\n"
            "display(row_pct)\n"
            "print('\\nColumn percentages:')\n"
            "display(col_pct)"
        ),
    )


def _build_time_trend(ctx, spec):
    time_col = _pick_time_col(ctx)
    value_col = _pick_numeric(ctx)
    if time_col is None or value_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request looks like a time-based trend analysis.",
        "This summarizes temporal patterns descriptively; any causal or forecasting claim would need a separate model.",
        (
            f"time_col = {time_col!r}\n"
            f"value_col = {value_col!r}\n"
            "temp = data.copy()\n"
            "temp[time_col] = pd.to_datetime(temp[time_col], errors='coerce')\n"
            "temp = temp.dropna(subset=[time_col, value_col])\n"
            "trend = temp.groupby(time_col)[value_col].sum().reset_index().sort_values(time_col)\n"
            "display(trend.head(10))\n"
            "plt.figure(figsize=(10, 4))\n"
            "sns.lineplot(data=trend, x=time_col, y=value_col, marker='o')\n"
            "plt.xticks(rotation=45)\n"
            "plt.title(f'Time Trend of {value_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_distribution_plot(ctx, spec):
    target_col = _pick_numeric(ctx)
    if target_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is about a variable distribution.",
        "The histogram and KDE are descriptive visual summaries rather than inferential tests.",
        (
            f"target_col = {target_col!r}\n"
            "display(data[target_col].describe().to_frame().T)\n"
            "plt.figure(figsize=(8, 4))\n"
            "sns.histplot(data[target_col].dropna(), kde=True)\n"
            "plt.title(f'Distribution of {target_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_pairplot_numeric(ctx, spec):
    numeric_cols = ctx.matched_numeric_cols or ctx.numeric_cols[:4]
    if len(numeric_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request is for a broad visual scan of relationships among numeric variables.",
        "Pairplots are exploratory visuals and should be followed by targeted diagnostics or formal models.",
        (
            f"numeric_cols = {repr(numeric_cols[:4])}\n"
            "temp = data[numeric_cols].dropna()\n"
            "display(temp.describe().T)\n"
            "sns.pairplot(temp, corner=True, diag_kind='hist')\n"
            "plt.show()"
        ),
    )


def _build_boxplot_by_group(ctx, spec):
    group_col = _pick_categorical(ctx)
    value_col = _pick_numeric(ctx)
    if group_col is None or value_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is for a distribution comparison across groups.",
        "Boxplots are descriptive and useful for comparing spread, medians, and outliers across groups.",
        (
            f"group_col = {group_col!r}\n"
            f"value_col = {value_col!r}\n"
            "summary = data.groupby(group_col)[value_col].agg(['count', 'mean', 'median', 'std']).reset_index()\n"
            "display(summary)\n"
            "plt.figure(figsize=(9, 4))\n"
            "sns.boxplot(data=data, x=group_col, y=value_col)\n"
            "plt.xticks(rotation=45)\n"
            "plt.title(f'{value_col} by {group_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_scatter_regression_plot(ctx, spec):
    numeric_cols = _pick_two_numeric(ctx)
    if len(numeric_cols) < 2:
        return None
    return _wrap_tool(
        spec,
        "The request is for a visual relationship between two numeric variables.",
        "The fitted line is a descriptive linear trend and should not be over-interpreted without diagnostics.",
        (
            f"x_col = {numeric_cols[0]!r}\n"
            f"y_col = {numeric_cols[1]!r}\n"
            "temp = data[[x_col, y_col]].dropna().astype(float)\n"
            "print(temp[[x_col, y_col]].corr())\n"
            "plt.figure(figsize=(7, 5))\n"
            "sns.regplot(data=temp, x=x_col, y=y_col, scatter_kws={'alpha': 0.7})\n"
            "plt.title(f'{y_col} vs {x_col}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_ols_regression(ctx, spec):
    target_col = _pick_numeric(ctx)
    if target_col is None:
        return None
    feature_cols = [col for col in ctx.numeric_cols if col != target_col][:6]
    if not feature_cols:
        return None
    return _wrap_tool(
        spec,
        "The request is a linear regression task and numeric predictors are available.",
        "OLS assumes independent observations, approximate linearity, homoscedastic residuals, and limited multicollinearity.",
        (
            "import statsmodels.api as sm\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
            f"target_col = {target_col!r}\n"
            f"feature_cols = {repr(feature_cols)}\n"
            "temp = data[feature_cols + [target_col]].dropna().copy()\n"
            "X = sm.add_constant(temp[feature_cols], has_constant='add')\n"
            "y = temp[target_col].astype(float)\n"
            "model = sm.OLS(y, X).fit()\n"
            "print(model.summary())\n"
            "vif = pd.DataFrame({\n"
            "    'feature': X.columns,\n"
            "    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n"
            "})\n"
            "display(vif)\n"
            "fitted = model.fittedvalues\n"
            "residuals = model.resid\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "sns.scatterplot(x=fitted, y=residuals, ax=axes[0])\n"
            "axes[0].axhline(0, color='red', linestyle='--')\n"
            "axes[0].set_title('Residuals vs Fitted')\n"
            "sns.histplot(residuals, kde=True, ax=axes[1])\n"
            "axes[1].set_title('Residual Distribution')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    )


def _build_logistic_regression(ctx, spec):
    target_candidates = ctx.matched_cols + ctx.categorical_cols + ctx.numeric_cols
    target_col = None
    for col in target_candidates:
        if col not in ctx.columns:
            continue
        nunique = ctx.dataframe[col].dropna().nunique()
        if 1 < nunique <= 2:
            target_col = col
            break
    if target_col is None:
        return None
    return _wrap_tool(
        spec,
        "The request is a binary classification or logistic regression task.",
        "This uses a train-test split and assumes the binary target is meaningfully encoded with predictors that do not leak future information.",
        (
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            f"target_col = {target_col!r}\n"
            "candidate_numeric = [col for col in data.select_dtypes(include=['number']).columns if col != target_col][:8]\n"
            "candidate_categorical = [col for col in data.select_dtypes(exclude=['number']).columns if col != target_col and data[col].nunique(dropna=True) <= 12][:4]\n"
            "feature_cols = candidate_numeric + candidate_categorical\n"
            "temp = data[feature_cols + [target_col]].dropna(subset=[target_col]).copy()\n"
            "y = temp[target_col].astype('category').cat.codes\n"
            "X = temp[feature_cols]\n"
            "numeric_features = [col for col in candidate_numeric if col in X.columns]\n"
            "categorical_features = [col for col in candidate_categorical if col in X.columns]\n"
            "preprocess = ColumnTransformer([\n"
            "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),\n"
            "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),\n"
            "])\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\n"
            "model = Pipeline([('preprocess', preprocess), ('clf', LogisticRegression(max_iter=1000))])\n"
            "model.fit(X_train, y_train)\n"
            "pred = model.predict(X_test)\n"
            "proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None\n"
            "print('Accuracy:', accuracy_score(y_test, pred))\n"
            "print('\\nClassification report:')\n"
            "print(classification_report(y_test, pred))\n"
            "print('\\nConfusion matrix:')\n"
            "print(confusion_matrix(y_test, pred))\n"
            "if proba is not None:\n"
            "    print('ROC AUC:', roc_auc_score(y_test, proba))"
        ),
    )


def _contains_any(message, terms):
    return any(term in message for term in terms)


TOOL_REGISTRY = [
    AnalysisToolSpec(
        name="dataset_overview",
        category="descriptive_statistics",
        description="Inspect dataset size, dtypes, missingness, duplicates, and quick summaries.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            [
                "how many rows", "how many columns", "read the dataset", "dataset overview",
                "preview", "head()", "schema", "show the data",
            ],
        ),
        builder=_build_dataset_overview,
    ),
    AnalysisToolSpec(
        name="descriptive_statistics",
        category="descriptive_statistics",
        description="Produce descriptive statistics for numeric and categorical columns.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["summary statistics", "descriptive statistics", "describe", "mean", "median", "std", "quantile", "percentile"],
        ),
        builder=_build_descriptive_statistics,
    ),
    AnalysisToolSpec(
        name="missingness_profile",
        category="descriptive_statistics",
        description="Summarize missing values and visualize missingness patterns.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["missingness", "missing value", "missing values", "null", "na", "completeness", "data quality"],
        ),
        builder=_build_missingness_profile,
    ),
    AnalysisToolSpec(
        name="outlier_screening",
        category="descriptive_statistics",
        description="Screen numeric variables for outliers using IQR and z-score heuristics.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["outlier", "outliers", "anomaly", "anomalies", "extreme value", "unusual values"],
        ),
        builder=_build_outlier_screening,
    ),
    AnalysisToolSpec(
        name="normality_check",
        category="hypothesis_testing",
        description="Check distributional normality with tests and plots.",
        matcher=lambda ctx: _contains_any(ctx.message, ["normality", "normality test", "qq plot", "normal assumption", "shapiro"]),
        builder=_build_normality_check,
    ),
    AnalysisToolSpec(
        name="one_sample_t_test",
        category="hypothesis_testing",
        description="Compare a sample mean against a fixed benchmark.",
        matcher=lambda ctx: _contains_any(ctx.message, ["one sample t-test", "one-sample t-test", "compare mean to", "different from"]) and _extract_first_number(ctx.user_message) is not None,
        builder=_build_one_sample_t_test,
    ),
    AnalysisToolSpec(
        name="paired_t_test",
        category="hypothesis_testing",
        description="Run a paired t-test for matched measurements.",
        matcher=lambda ctx: _contains_any(ctx.message, ["paired t-test", "paired test", "before after", "pre post", "matched pairs"]),
        builder=_build_paired_t_test,
    ),
    AnalysisToolSpec(
        name="variance_homogeneity",
        category="hypothesis_testing",
        description="Test whether group variances are similar using Levene and Bartlett tests.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["equal variance", "variance equality", "homogeneity of variance", "homoscedasticity", "levene", "bartlett"],
        ),
        builder=_build_variance_homogeneity,
    ),
    AnalysisToolSpec(
        name="wilcoxon_signed_rank",
        category="hypothesis_testing",
        description="Run a Wilcoxon signed-rank test for paired nonparametric comparisons.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["wilcoxon", "signed-rank", "signed rank", "paired nonparametric"],
        ),
        builder=_build_wilcoxon_signed_rank,
    ),
    AnalysisToolSpec(
        name="mann_whitney_u_test",
        category="hypothesis_testing",
        description="Run a Mann-Whitney U test for two independent groups.",
        matcher=lambda ctx: _contains_any(ctx.message, ["mann-whitney", "mann whitney", "rank-sum", "nonparametric"]),
        builder=_build_mann_whitney,
    ),
    AnalysisToolSpec(
        name="one_way_anova",
        category="hypothesis_testing",
        description="Compare means across more than two groups with one-way ANOVA.",
        matcher=lambda ctx: _contains_any(ctx.message, ["anova", "one-way anova", "compare more than two groups"]),
        builder=_build_anova,
    ),
    AnalysisToolSpec(
        name="kruskal_wallis",
        category="hypothesis_testing",
        description="Run a Kruskal-Wallis test for nonparametric multi-group comparisons.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["kruskal", "kruskal-wallis", "kruskal wallis", "nonparametric anova"],
        ),
        builder=_build_kruskal_wallis,
    ),
    AnalysisToolSpec(
        name="chi_square_test",
        category="hypothesis_testing",
        description="Test association between two categorical variables.",
        matcher=lambda ctx: _contains_any(ctx.message, ["chi-square", "chi square", "contingency", "categorical association"]),
        builder=_build_chi_square,
    ),
    AnalysisToolSpec(
        name="fisher_exact_test",
        category="hypothesis_testing",
        description="Run Fisher's exact test for 2x2 categorical tables.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["fisher exact", "fisher's exact", "exact test", "2x2 table"],
        ),
        builder=_build_fisher_exact,
    ),
    AnalysisToolSpec(
        name="two_proportion_test",
        category="hypothesis_testing",
        description="Compare success proportions between two groups with a z-test.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["two proportion", "proportion test", "compare proportions", "conversion rate", "success rate"],
        ),
        builder=_build_two_proportion_test,
    ),
    AnalysisToolSpec(
        name="two_sample_t_test",
        category="hypothesis_testing",
        description="Compare the means of two independent groups.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["t-test", "independent t-test", "compare means", "difference in means", "significant difference between groups"],
        ),
        builder=_build_two_sample_t_test,
    ),
    AnalysisToolSpec(
        name="correlation_matrix",
        category="relationship_analysis",
        description="Compute and visualize a correlation matrix.",
        matcher=lambda ctx: _contains_any(ctx.message, ["correlation", "pearson", "spearman", "relationship between numeric variables"]),
        builder=_build_correlation,
    ),
    AnalysisToolSpec(
        name="time_trend_plot",
        category="visualization",
        description="Aggregate and visualize a metric over time.",
        matcher=lambda ctx: _contains_any(ctx.message, ["trend", "time series", "monthly", "daily", "over time", "line chart", "line plot"]),
        builder=_build_time_trend,
    ),
    AnalysisToolSpec(
        name="distribution_plot",
        category="visualization",
        description="Show a histogram and density estimate for a numeric variable.",
        matcher=lambda ctx: _contains_any(ctx.message, ["distribution", "histogram", "hist", "density", "kde"]),
        builder=_build_distribution_plot,
    ),
    AnalysisToolSpec(
        name="pairplot_numeric",
        category="visualization",
        description="Create a pairplot for a small set of numeric variables.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["pairplot", "pair plot", "scatter matrix", "visualize numeric relationships"],
        ),
        builder=_build_pairplot_numeric,
    ),
    AnalysisToolSpec(
        name="boxplot_by_group",
        category="visualization",
        description="Compare numeric distributions across groups with a boxplot.",
        matcher=lambda ctx: _contains_any(ctx.message, ["boxplot", "box plot", "outlier by group"]),
        builder=_build_boxplot_by_group,
    ),
    AnalysisToolSpec(
        name="scatter_regression_plot",
        category="visualization",
        description="Show a scatter plot with a fitted regression line.",
        matcher=lambda ctx: _contains_any(ctx.message, ["scatter plot", "scatter", "regression line", "relationship plot"]),
        builder=_build_scatter_regression_plot,
    ),
    AnalysisToolSpec(
        name="grouped_summary",
        category="descriptive_statistics",
        description="Summarize a numeric variable by group and visualize the comparison.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["group by", "by region", "by category", "comparison", "per region", "per category", "bar chart", "bar plot"],
        ),
        builder=_build_grouped_summary,
    ),
    AnalysisToolSpec(
        name="categorical_frequency",
        category="descriptive_statistics",
        description="Summarize counts and proportions for one categorical variable.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["frequency", "counts", "countplot", "class balance", "category distribution"],
        ),
        builder=_build_categorical_frequency,
    ),
    AnalysisToolSpec(
        name="crosstab_summary",
        category="descriptive_statistics",
        description="Build counts and percentages for two categorical variables.",
        matcher=lambda ctx: _contains_any(
            ctx.message,
            ["crosstab", "cross tab", "cross-tab", "contingency table", "row percentage", "column percentage"],
        ),
        builder=_build_crosstab_summary,
    ),
    AnalysisToolSpec(
        name="ols_regression",
        category="regression_modeling",
        description="Fit an ordinary least squares regression with diagnostics.",
        matcher=lambda ctx: _contains_any(ctx.message, ["regression", "ols", "linear model", "linear regression"])
        and not _contains_any(ctx.message, ["logistic regression", "binary classification", "classify", "classification model", "predict class"]),
        builder=_build_ols_regression,
    ),
    AnalysisToolSpec(
        name="logistic_regression",
        category="regression_modeling",
        description="Fit a logistic regression model for binary classification.",
        matcher=lambda ctx: _contains_any(ctx.message, ["logistic regression", "binary classification", "classify", "classification model", "predict class"]),
        builder=_build_logistic_regression,
    ),
]


def route_builtin_skill(user_message, dataframe):
    if dataframe is None:
        return None

    ctx = _build_context(user_message, dataframe)
    for spec in TOOL_REGISTRY:
        if not spec.matcher(ctx):
            continue
        tool = spec.builder(ctx, spec)
        if tool:
            return tool
    return None


def embed_skill_code(skill):
    if skill is None:
        return None
    lines = [
        f"# Built-in statbot tool: {skill['name']}",
        f"# Category: {skill.get('category', 'general')}",
        f"# Rationale: {skill['rationale']}",
    ]
    if skill.get("assumptions"):
        lines.append(f"# Assumptions: {skill['assumptions']}")
    return (
        "\n".join(lines)
        + "\n```python\n"
        + skill["code"]
        + "\n```"
    )


def summarize_request(user_message, skill_name=None):
    clean = re.sub(r"\s+", " ", user_message).strip()
    if skill_name:
        return f"Handled request '{clean[:120]}' with built-in tool '{skill_name}'."
    return f"Handled request '{clean[:120]}' with dynamic code generation."
