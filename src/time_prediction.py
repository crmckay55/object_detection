# time_prediction.py
# Chris McKay
# v1.0 2021-11-14
# a little class to predict time to finish based on number of items and elapsed time

from datetime import datetime, timedelta

class TimePredictor():

    def __init__(self, item_count):
        self.total_items = item_count
        self.start_time = datetime.now()

    def reset(self):
        self.start_time = datetime.now()

    def set_items(self, total_items):
        self.total_items = total_items

    def predict(self, current_item):
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        avg_time_per_item = current_item / elapsed_time.total_seconds()
        prediction = current_time + timedelta(seconds=self.total_items / avg_time_per_item)
        prediction = prediction.strftime('%H:%M')
        print(f'Average processing is {round(avg_time_per_item, 3)} items per minute.  Predicting finished at {prediction}.')