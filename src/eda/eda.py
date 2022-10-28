import pandas as pd
import logging
from matplotlib import pyplot as plt


class EDA:
    def __init__(self, data, show_plot=True):
        self.data = data

        if show_plot:
            plt.ion()
        else:
            plt.off()

    def run(self):
        # total_sales_after_discount, discount_percent by weekday/weekend
        isweekend = self.data.groupby('is_weekend')
        isweekend_mean = isweekend.aggregate('mean')
        fig_isweekend, ax_isweekend = plt.subplots(2, 2)
        ax_isweekend = ax_isweekend.flatten()
        ax_isweekend[0].bar(isweekend_mean.index.astype(str), isweekend_mean['total_sales_after_discount'])
        ax_isweekend[0].set(title='total_sales_after_discount', ylabel='USD')
        self.data[['is_weekend','total_sales_after_discount']].boxplot(by='is_weekend', ax=ax_isweekend[1])
        ax_isweekend[2].bar(isweekend_mean.index.astype(str), isweekend_mean['discount_percent'])
        ax_isweekend[2].set(title='discount_percent', ylabel='USD')
        self.data[['is_weekend','discount_percent']].boxplot(by='is_weekend', ax=ax_isweekend[3])
        for key, value in isweekend:
            logging.info(f"When is_weekend={key}, Median total_sales_after_discount={value['total_sales_after_discount'].median():.2f}")

        # total_sales_after_discount, discount_percent by day of week
        dayofweek = self.data.groupby('day_of_week')
        dayofweek_mean = self.data.groupby('day_of_week').aggregate('mean')
        fig_dayofweek, ax_dayofweek = plt.subplots(2, 2)
        ax_dayofweek = ax_dayofweek.flatten()
        ax_dayofweek[0].bar(dayofweek_mean.index, dayofweek_mean['total_sales_after_discount'])
        ax_dayofweek[0].set(title='total_sales_after_discount', ylabel='USD')
        self.data[['day_of_week','total_sales_after_discount']].boxplot(by='day_of_week', ax=ax_dayofweek[1])
        ax_dayofweek[2].bar(dayofweek_mean.index, dayofweek_mean['discount_percent'])
        ax_dayofweek[2].set(title='discount_percent', ylabel='USD')
        self.data[['day_of_week','discount_percent']].boxplot(by='day_of_week', ax=ax_dayofweek[3])
        for key, value in isweekend:
            logging.info(f"When day_of_week={key}, Median discount_percent={value['discount_percent'].median():.2f}")

        # Average spending per transaction
        transaction = self.data.groupby('transaction_id').aggregate('mean')
        fig_transaction, ax_transaction = plt.subplots()
        transaction[['total_sales_after_discount']].boxplot(ax=ax_transaction)

        # Average spending per member
        member = self.data.groupby('member_id_masked').aggregate('mean')
        fig_member, ax_member = plt.subplots()
        member[['total_sales_after_discount']].boxplot(ax=ax_member)

        pass
