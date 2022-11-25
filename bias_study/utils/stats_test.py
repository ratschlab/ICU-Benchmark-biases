import pandas as pd
import scipy


def run_k_sample_anderson(df, var, hue, cats=None):
    """Run k-sample Anderson-Darling test

    Args:
        df (Pandas DataFrame): dataframe
        var (string): continuous variable
        hue (string): categorical variable defining the groups
        cats (list, optional): list to precise the categories of hue to look at. Defaults to None.
    """
    if not cats:
        cats = df[df[hue].notna()][hue].unique()
    samples = [df[df[hue] == cat][var] for cat in cats]
    return scipy.stats.anderson_ksamp(samples)


def run_2_ks(df, var, hue, group1, group2):
    """Run 2-sample Kolmogorov-Smirnov test

    Args:
        df (Pandas DataFrame): dataframe
        var (string): continuous variable
        hue (string): categorical variable defining the groups
        group1 (string or bool or int): value of hue defining group1
        group2 (string or bool or int): value of hue defining group2
    """
    data1 = df[df[hue] == group1][var]
    data2 = df[df[hue] == group2][var]
    return scipy.stats.ks_2samp(data1, data2)


def run_mann_whitney_u(df, var, hue, group1, group2, hyp="two-sided"):
    """Run Mann-Whitney U test
    Args:
        df (Pandas DataFrame): dataframe
        var (string): continuous variable
        hue (string): categorical variable defining the groups
        group1 (string or bool or int): value of hue defining group1
        group2 (string or bool or int): value of hue defining group2
        hyp (string, optional): define the null hypothesis
    """
    data1 = df[df[hue] == group1][var]
    data2 = df[df[hue] == group2][var]
    return scipy.stats.mannwhitneyu(data1, data2, alternative=hyp, nan_policy="omit")


def run_k_sample_kruskal(df, var, hue, cats=None):
    """Run k-sample Kruskal-Wallis test

    Args:
        df (Pandas DataFrame): dataframe
        var (string): continuous variable
        hue (string): categorical variable defining the groups
        cats (list, optional): list to precise the categories of hue to look at. Defaults to None."""
    if not cats:
        cats = df[df[hue].notna()][hue].unique()
    samples = [df[df[hue] == cat][var].dropna() for cat in cats]
    return scipy.stats.kruskal(*samples, nan_policy="omit")


def run_2_kruskal(group1, group2, var):
    """Run 2-sample Kruskal-Wallis test

    Args:
        group1 (Pandas dataframe): dataframe for the first group
        group2 (Pandas dataframe): dataframe for the second group
        var (string): continuous variable
    """
    data1 = group1[var]
    data2 = group2[var]
    return scipy.stats.kruskal(data1, data2, nan_policy="omit")


def run_chi_square_independence_test(df, var1, var2, filter1=None, filter2=None):
    """Run Chi-Square test of independence.

    Args:
        df (Pandas Dataframe): dataframe
        var1 (string): first categorical variable
        var2 (string): second categorical variable
        filter1 (list, optional): list of categories of var1 to include. Defaults to None.
        filter2 (list, optional): list of categories of var2 to include. Defaults to None.
    """
    if filter1:
        df = df[df[var1].isin(filter1)]
    if filter2:
        df = df[df[var2].isin(filter2)]
    table = pd.crosstab(df[var1], df[var2])
    return scipy.stats.contingency.chi2_contingency(table)


def run_chi_square_ind_subgroup(df, group1, group2, var, cats=None):
    """Run Chi-Square test of independence on predefined subgoups

    Args:
        df (Pandas dataframe): dataframe
        group1 (Pandas dataframe): dataframe for the first group
        group2 (Pandas dataframe): dataframe for the second group
        var (string): categorical variable
        cats (list, optional): list of categories of var to include. Defaults to None.
    """
    if not cats:
        cats = df[df[var].notna()][var].unique()
    table = [
        [len(group1[group1[var] == cat]) for cat in cats],
        [len(group2[group2[var] == cat]) for cat in cats],
    ]
    return scipy.stats.contingency.chi2_contingency(table)


def run_spearman_r(df, var1, var2, hyp="two-sided"):
    """Run Spearman's R correlation.

    Args:
        df (Pandas dataframe): dataframe
        var1 (string): first continuous variable
        var2 (string): second continuous variable
        hyp (str, optional): hypothesis to test. Defaults to 'two-sided'.
    """
    return scipy.stats.spearmanr(df[var1], df[var2], nan_policy="omit", alternative=hyp)


def run_pearson_r(df, var1, var2, hyp="two-sided"):
    """Run Pearson's R correlation.

    Args:
        df (Pandas dataframe): dataframe
        var1 (string): first continuous variable
        var2 (string): second continuous variable
        hyp (str, optional): hypothesis to test. Defaults to 'two-sided'.
    """
    return scipy.stats.pearsonr(df[var1], df[var2], alternative=hyp)


def run_linear_reg(df, var1, var2, hyp="two-sided"):
    """Run linear regression of var2 against var1.

    Args:
        df (Pandas dataframe): dataframe
        var1 (string): first continuous variable
        var2 (string): second continuous variable
        hyp (str, optional): hypothesis to test. Defaults to 'two-sided'.
    """
    return scipy.stats.linregress(df[var1], df[var2], alternative=hyp)
