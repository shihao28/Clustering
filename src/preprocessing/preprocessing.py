import pandas as pd
import logging
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed


class Preprocessing:
    """
    This class is use-case dependent
    """
    def __init__(self, data):
        self.data = data

    def _tranformation(self, key, value):
        # Aggregate recored by transaction date
        value = value.groupby('transaction_date', as_index=False).sum()

        # Create new features
        value['total_sales_after_discount'] = value['sales_value_usd'] * value['sales_unit']
        value['discount_percent'] = abs(value['original_price_usd'] - value['sales_value_usd']) / value['original_price_usd'] * 100
        value["day_of_week"] = value['transaction_date'].dt.dayofweek
        value['is_weekend'] = value["day_of_week"] > 4

        # Compute statistics
        customer = dict(member_id_masked=key)
        for feature in ['sales_value_usd', 'original_price_usd', 'discount_amount_usd', 'sales_unit', 'total_sales_after_discount', 'discount_percent']:
            customer[f'{feature}_min'] = value[feature].min()
            customer[f'{feature}_mean'] = value[feature].mean()
            customer[f'{feature}_median'] = value[feature].median()
            customer[f'{feature}_max'] = value[feature].max()

        # recency, frequency, monetary
        snapshot_date = max(self.data['transaction_date']) + datetime.timedelta(days=1)
        rfm = value.agg({
            'transaction_date': lambda x: (snapshot_date - x.max()).days,
            'transaction_id': 'count',
            'total_sales_after_discount': 'sum'}).to_frame()
        rfm.index = ['recency_in_days', 'frequency', 'monetary']
        customer.update(rfm.T.to_dict())

        # #transaction on weekday, weekends
        for i in range(7):
            customer[f'day{i}_transaction'] = (value['day_of_week']==i).sum()
        weekend_transaction_count = value['is_weekend'].sum()
        customer['weekend_transaction_count_percent'] = weekend_transaction_count/len(value)*100
        customer['weekday_transaction_count_percent'] = 100 - customer['weekend_transaction_count_percent']

        return pd.DataFrame(customer, index=[0])

    def run(self):
        logging.info("Preprocessing...")
        # Merge
        # self.data = pd.merge(self.data_sales, self.data_segment, 'left', 'member_id_masked')

        # Convert to dt format
        self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'], format='%Y%m%d')

        # Check dtypes and missing val
        self.data.info()

        # fill na with 0
        self.data.fillna(0, inplace=True)

        # Keep only sales_value_usd that is larger than 0
        self.data = self.data.loc[self.data['sales_value_usd'] > 0, ]
        # sales_unit should be larger than 0
        # self.data = self.data.loc[self.data['sales_unit'] > 0, ]
        # original_price_usd should be larger than 0
        # self.data = self.data.loc[self.data['original_price_usd'] > 0, ]

        # Create new features
        self.data['total_sales_after_discount'] = self.data['sales_value_usd'] * self.data['sales_unit']

        # data transformation
        customers = Parallel(n_jobs=-1)(delayed(self._tranformation)(key, value) for key, value in tqdm(self.data.groupby('member_id_masked')))
        customers = pd.concat(customers, axis=0, ignore_index=True)
        customers = pd.merge(customers, self.data[['member_id_masked', 'customer_type']], 'left', 'member_id_masked')

        # create rfm score
        customers["recency_score"] = pd.qcut(customers["recency_in_days"], 5, labels=[5, 4, 3, 2, 1])
        customers["frequency_score"] = pd.qcut(customers["frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        customers["monetary_score"] = pd.qcut(customers["monetary"], 5, labels=[1, 2, 3, 4, 5])
        customers["rfm_score"] = customers["recency_score"].astype(int) +\
            customers["frequency_score"].astype(int) + customers["monetary_score"].astype(int)

        # customers.to_csv('data/03_primary/customers.csv', index=False)

        return customers
