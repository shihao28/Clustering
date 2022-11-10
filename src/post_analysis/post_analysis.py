import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import logging


class PostAnalysis:
    """
    This class is use-case dependent
    """
    def __init__(self, data, best_clusterer, y, show_plot=True):
        self.data = data.drop(['member_id_masked', 'customer_type'], axis=1)
        self.best_clusterer = best_clusterer
        self.data['predicted_cluster'] = y
        self.numeric_features_names = [
            'sales_value_usd_mean', 'original_price_usd_mean', 'discount_amount_usd_mean',
            'total_sales_after_discount_mean', 'discount_percent_mean', 'frequency',
            'monetary', 'day0_transaction', 'day1_transaction', 'day2_transaction',
            'day3_transaction', 'day4_transaction', 'day5_transaction',
            'day6_transaction', 'sales_unit_mean', 'weekend_transaction_count_percent',
            'weekday_transaction_count_percent', 'recency_in_days', 
            ]
        self.category_features_names = [
            'recency_score', 'frequency_score',
            'monetary_score', 'rfm_score']
        if show_plot:
            plt.ion()
        else:
            plt.off()
        self.num_plot_per_fig = 6

    def run(self):
        logging.info("Post Analysis...")
        # Analyse numerical features
        cluster_mean_plot_all = []
        cluster_boxplot_plot_all = []
        for i, column in enumerate(self.numeric_features_names):
            if i % self.num_plot_per_fig == 0:
                fig_cluster_mean, ax_cluster_mean = plt.subplots(2, 3)
                ax_cluster_mean = ax_cluster_mean.flatten()
                fig_cluster_boxplot, ax_cluster_boxplot = plt.subplots(2, 3)
                ax_cluster_boxplot = ax_cluster_boxplot.flatten()
            data_mean = self.data.groupby('predicted_cluster').aggregate('mean')
            data_mean[[column]].plot(kind='bar', ax=ax_cluster_mean[i%self.num_plot_per_fig])
            self.data[['predicted_cluster', column]].boxplot(by='predicted_cluster', ax=ax_cluster_boxplot[i%self.num_plot_per_fig])
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.numeric_features_names)-1:
                cluster_mean_plot_all.append(fig_cluster_mean)
                cluster_boxplot_plot_all.append(ax_cluster_mean)
                plt.tight_layout()

        # Analyse category features
        cluster_category_features_plot_all = []
        for i, column in enumerate(self.category_features_names):
            if i % self.num_plot_per_fig == 0:
                fig_cluster_category_features, ax_cluster_category_features = plt.subplots(2, 3)
                ax_cluster_category_features = ax_cluster_category_features.flatten()
            X = self.data.groupby(['predicted_cluster', column]).size().to_frame('occurences').reset_index()
            sns.barplot(x='predicted_cluster', y='occurences', hue=column, data=X, ax=ax_cluster_category_features[i%self.num_plot_per_fig])
            if (i + 1) % self.num_plot_per_fig == 0 or i == len(self.category_features_names)-1:
                cluster_category_features_plot_all.append(fig_cluster_category_features)
                plt.tight_layout()

        return tuple(cluster_mean_plot_all + cluster_boxplot_plot_all + cluster_category_features_plot_all)
