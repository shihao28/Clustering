import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler


class EDA:
    def __init__(self, data, scale=True, show_plot=True):
        self.data = data.drop(['member_id_masked', 'customer_type'], axis=1)
        self.scale = scale
        self.num_plot_per_fig = 6
        if show_plot:
            plt.ion()
        else:
            plt.ioff()

    def _univariate_analysis(self):
        # numeric features
        numeric_features_stats = self.data.agg([
            "min", "max", "mean", "median", "std",
            "skew", "kurtosis"])
        numeric_features_quantile = self.data.quantile([0.25, 0.75])
        numeric_features_stats = pd.concat(
            [numeric_features_stats, numeric_features_quantile], 0)
        numeric_features_stats.sort_values(
            by="std", axis=1, ascending=True, inplace=True)
        numeric_features_stats = numeric_features_stats.round(4)
        # numeric_features_stats_fig, numeric_features_stats_ax = plt.subplots()
        # numeric_features_stats_ax.axis('tight')
        # numeric_features_stats_ax.axis('off')
        # numeric_features_stats_ax.table(
        #     cellText=numeric_features_stats.values,
        #     colLabels=numeric_features_stats.columns,
        #     rowLabels=numeric_features_stats.index, loc='center', fontsize=20)
        # numeric_features_stats.to_csv('data/04_feature/stat.csv')

        # Kde plot
        kdeplot_all = []
        for i, (name, column) in enumerate(self.data.iteritems()):
            if i % self.num_plot_per_fig == 0:
                kdeplot_fig, kdeplot_ax = plt.subplots(2, 3)
                kdeplot_ax = kdeplot_ax.flatten()
            column.plot.kde(
                ax=kdeplot_ax[i%self.num_plot_per_fig], secondary_y=True, title=name)
            # ax_tmp.text(0.5, 0.5, "test", fontsize=22)
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.data.columns)-1:
                kdeplot_all.append(kdeplot_fig)
                plt.tight_layout()

        # Boxplot
        boxplot_all = []
        for i, (name, column) in enumerate(self.data.iteritems()):
            if i % self.num_plot_per_fig == 0:
                boxplot_fig, boxplot_ax = plt.subplots(2, 3)
                boxplot_ax = boxplot_ax.flatten()
            boxplot_ax[i%self.num_plot_per_fig].boxplot(column, 0, 'gD')
            boxplot_ax[i%self.num_plot_per_fig].set_title(f"{name}")
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.data.columns)-1:
                boxplot_all.append(boxplot_fig)
                plt.tight_layout()

        return numeric_features_stats, kdeplot_all, boxplot_all

    def _bivariate_analysis(self):
        # Numerical vs Numerical
        # correlation plot
        corr_matrix = self.data.corr(method="pearson")
        corr_fig, corr_ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=corr_ax, annot=True)
        corr_ax.set(title="Correlation Matrix (Pearson)")

        num_vs_num_plot_all = []
        # columns_comb = list(combinations(self.data.columns, 2))
        # for i, (column_a, column_b) in enumerate(columns_comb):
        #     if i % self.num_plot_per_fig == 0:
        #         num_vs_num_plot_fig, num_vs_num_plot_ax = plt.subplots(3, 2)
        #         num_vs_num_plot_ax = num_vs_num_plot_ax.flatten()
        #     num_vs_num_plot_ax[i%self.num_plot_per_fig].scatter(
        #         x=self.data[column_a], y=self.data[column_b])
        #     num_vs_num_plot_ax[i%self.num_plot_per_fig].set(
        #         xlabel=column_a, ylabel=column_b,
        #         title=f"Plot of {column_b} vs {column_a}")
        #     if (i + 1) % self.num_plot_per_fig == 0 or i == len(columns_comb)-1:
        #         num_vs_num_plot_all.append(num_vs_num_plot_fig)
        #         plt.tight_layout()

        return corr_fig, num_vs_num_plot_all

    def _top_segment_analysis(self):
        # Identify top segment customer
        # Top segment customer is above 0.97 quantile
        quantiles = np.arange(90, 100) / 100
        monetary_vals = []
        segment_count = []
        frequencies_mean = []
        frequencies_median = []
        recencies_mean = []
        recencies_median = []
        for quantile in quantiles:
            monetary_val = self.data['monetary'].quantile(quantile)
            monetary_vals.append(monetary_val)

            data_tmp = self.data.loc[self.data['monetary'] >= monetary_val,]
            segment_count.append(len(data_tmp))
            frequencies_mean.append(data_tmp['frequency'].mean())
            frequencies_median.append(data_tmp['frequency'].median())
            recencies_mean.append(data_tmp['recency_in_days'].mean())
            recencies_median.append(data_tmp['recency_in_days'].median())

        # Plot qunatile against monetary_vals and segment_count
        fig_monetary_quantile, ax_monetary_quantile = plt.subplots()
        ax_monetary_quantile.plot(quantiles, monetary_vals)
        ax_monetary_quantile2=ax_monetary_quantile.twinx()
        ax_monetary_quantile2.plot(quantiles, segment_count, color='orange')
        ax_monetary_quantile.set(xlabel='quantile', ylabel='monetary')
        ax_monetary_quantile2.set(ylabel='customer_count')
        ax_monetary_quantile.legend(['monetary'])
        ax_monetary_quantile2.legend(['customer_count'])

        # Plot quantile aginst frequency
        fig_frequency_quantile, ax_frequency_quantile = plt.subplots()
        ax_frequency_quantile.plot(quantiles, frequencies_mean)
        ax_frequency_quantile2 = ax_frequency_quantile.twinx()
        ax_frequency_quantile2.plot(quantiles, frequencies_median, color='orange')
        ax_frequency_quantile.set(xlabel='quantile', ylabel='frequency_mean')
        ax_frequency_quantile2.set(ylabel='frequency_median')
        ax_frequency_quantile.legend(['frequency_mean'])
        ax_frequency_quantile2.legend(['frequency_median'])

        # Plot quantile aginst recencies
        fig_recency_quantile, ax_recency_quantile = plt.subplots()
        ax_recency_quantile.plot(quantiles, recencies_mean)
        ax_recency_quantile2 = ax_recency_quantile.twinx()
        ax_recency_quantile2.plot(quantiles, recencies_median, color='orange')
        ax_recency_quantile.set(xlabel='quantile', ylabel='recency_mean')
        ax_recency_quantile2.set(ylabel='recency_median')
        ax_recency_quantile.legend(['recency_mean'])
        ax_recency_quantile2.legend(['recency_median'])

        return fig_monetary_quantile, fig_frequency_quantile, fig_recency_quantile

    def run(self):
        # Apply scaling
        self.data[['discount_amount_usd_min', 'discount_amount_usd_median', 'discount_amount_usd_mean', 'discount_amount_usd_max']]=\
            np.log(self.data[['discount_amount_usd_min', 'discount_amount_usd_median', 'discount_amount_usd_mean', 'discount_amount_usd_max']] + 1)
        self.data[[
            'sales_value_usd_min', 'sales_value_usd_median', 'sales_value_usd_mean', 'sales_value_usd_max',
            'original_price_usd_min', 'original_price_usd_median', 'original_price_usd_mean', 'original_price_usd_max',
            'total_sales_after_discount_min', 'total_sales_after_discount_median', 'total_sales_after_discount_mean', 'total_sales_after_discount_max',
            'monetary', 'frequency']] = np.log(self.data[[
                'sales_value_usd_min', 'sales_value_usd_median', 'sales_value_usd_mean', 'sales_value_usd_max',
                'original_price_usd_min', 'original_price_usd_median', 'original_price_usd_mean', 'original_price_usd_max',
                'total_sales_after_discount_min', 'total_sales_after_discount_median', 'total_sales_after_discount_mean', 'total_sales_after_discount_max',
                'monetary', 'frequency']])
        self.data.iloc[:,:] = StandardScaler().fit_transform(self.data)

        # Univariate and bivariate analysis
        numeric_features_stats, kdeplot_all, boxplot_all = self._univariate_analysis()

        corr_fig, num_vs_num_plot_all = self._bivariate_analysis()

        # Top segment analysis
        fig_monetary_quantile, fig_frequency_quantile, fig_recency_quantile = self._top_segment_analysis()

        return numeric_features_stats, kdeplot_all, boxplot_all, corr_fig, num_vs_num_plot_all
