import pandas as pd
import numpy as np


def feature_stats(df, target):

    def basic_stats(df, df_stats):
        '''
        calculate extra basic stats of features including feature medians, 
        n-th largest/smallest values, number of unique/null values
        '''
        df_stats['median'] = df.median()
        for el in range(5, 21, 5):
            df_stats[str(
                el) + '-th_lar_val'] = df.apply(lambda col: col.nlargest(el).values[-1])
        for el in range(5, 21, 5):
            df_stats[str(
                el) + '-th_sma_val'] = df.apply(lambda col: col.nsmallest(el).values[-1])
        mask = df.isnull()
        df_stats['na_cnt'] = mask.sum()
        df_stats['nunique'] = df.nunique()
        return df_stats

    def corr_stats(df, df_stats):
        '''
        calculate correlation between x and y with simple transformations 
        include x^2, x^3, x^0.5, log(x), tanh(x), sigmoid(x)
        '''
        df_stats['corr(x,y)'] = df.corrwith(df.y).abs()
        df_stats['corr(x^2,y)'] = (df**2).corrwith(df.y).abs()
        df_stats['corr(x^3,y)'] = (df**3).corrwith(df.y).abs()
        df_stats['corr(x^.5,y)'] = (df**.5).corrwith(df.y).abs()
        df_stats['corr(logx,y)'] = np.log(
            df+1).corrwith(df.y).abs()
        df_stats['corr(tanhx,y)'] = np.tanh(
            df).corrwith(df.y).abs()
        df_stats['corr(sigx,y'] = (
            1 / (1 + np.exp(-df))).corrwith(df.y).abs()
        return df_stats

    def dependent_stats(df, df_stats, qt=[.99, .95, .90, .10, .05, .01]):
        '''
        calculate top and bottom quantile means for outlier detection
        '''
        for el in qt:
            if el > 0.5:
                mask = df >= df.quantile(el)
            else:
                mask = df <= df.quantile(el)
            df_stats[str(el) + '-qt_mean_y'] = mask.T @ df.y / mask.sum()
        return df_stats

    def zero_stats(df, df_stats):
        'check 0 value counts and means'
        zero_mask = df == 0 & ~df.isnull()
        df_stats['0_cnt'] = zero_mask.sum()
        df_stats['non_0_mean_x'] = df[~zero_mask].mean()
        df_stats['non_0_mean_y'] = ~zero_mask.T @ df.y / \
            (~zero_mask).sum()
        return df_stats

    def finalize(df_stats):
        df_stats.drop(columns=['count'], inplace=True)
        #order = ['mean', 'median', 'std', '0_cnt', 'na_cnt', 'nunique']
        order = ['mean', 'median', 'std', 'nunique']
        order += [el for el in df_stats.columns if el not in order]
        df_stats = df_stats[order]
        df_stats.drop(index=['y'], inplace=True)
        return df_stats

    df.rename(columns={target: 'y'}, inplace=True)
    df_stats = df.describe().T
    df_stats = basic_stats(df, df_stats)
    df_stats = corr_stats(df, df_stats)
    df_stats = dependent_stats(df, df_stats)
    #df_stats = zero_stats(df, df_stats)
    df_stats = finalize(df_stats)

    return df_stats
