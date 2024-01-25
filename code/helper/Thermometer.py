import pandas as pd
from datetime import datetime
import numpy as np

class Thermometer:
    """
    This class provides methods to read room and Viewpixx temperatures from specified files within a given date range.
    """


    @staticmethod
    def ReadRoomTemperature(filename, daterange=('1970-01-01', '2100-12-31')):
        """
        Reads room temperature data from a file within the specified date range.

        Parameters:
        - filename (str): Path to the file containing room temperature data.
        - daterange (tuple): Date range of interest in the format ('start_date', 'end_date') (default is ('1970-01-01', '2100-12-31')).

        Returns:
        pd.DataFrame: DataFrame containing datetime and temperature columns within the specified date range.
        """

        doi = [datetime.strptime(d, '%Y-%m-%d %H:%M') for d in daterange]
        t = pd.read_csv(filename, header=None, sep=' ', engine='c', skiprows=486)
        t.columns = ['xdt', 'temperature']    # no header, give columns a name
        t['datetime'] = pd.to_datetime(t['xdt'], format='%Y-%m-%d/%H:%M') # convert string to datetime object
        mask = (t['datetime'] >= doi[0]) & (t['datetime'] <= doi[1])  # restrict to date of interest
        return t.loc[mask][['datetime', 'temperature']]


    @staticmethod
    def ReadViewpixxTemperature(filename, daterange=('1970-01-01 00:00', '2100-12-31 23:59')):
        """
        Reads Viewpixx temperature data from a file within the specified date range.

        Parameters:
        - filename (str): Path to the file containing Viewpixx temperature data.
        - daterange (tuple): Date range of interest in the format ('start_date', 'end_date') (default is ('1970-01-01 00:00', '2100-12-31 23:59')).

        Returns:
        pd.DataFrame: DataFrame containing datetime and temperature columns for 8 channels within the specified date range.
        """

        dates = pd.read_csv(filename, header=None, sep=' ', skiprows=lambda i: i % 2, engine='c') # odd lines with date&time
        x = pd.DataFrame({'datetime': pd.to_datetime(dates.values[:, 0], format='%Y%m%d%H%M%S').values}) # into a DataFrame
        traw = pd.read_csv(filename, header=None, sep=' ', skiprows=lambda i: 1 - i % 2, engine='c') # even lines from viewpixx
        cc = np.arange(8, 85, 10)  # in a line, every 10th word starting at 8 is a temperature
        nn = ['t'+str(i) for i in range(len(cc))]
        for n, c in zip(nn, cc):   # the 8 channels of the backpanels readings
            x[n] = traw.iloc[:, c]
        doi = [datetime.strptime(d, '%Y-%m-%d %H:%M') for d in daterange] 
        mask = (x['datetime'] >= doi[0]) & (x['datetime'] <= doi[1])  # restrict to date of interest
        # ensure that t0-t7 are numbers
        cols = x.columns.drop('datetime')
        x[cols] = x[cols].apply(pd.to_numeric, errors='coerce')
        return x.loc[mask]
    