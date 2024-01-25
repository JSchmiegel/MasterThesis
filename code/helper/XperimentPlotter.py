import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import fnmatch
import os
from Thermometer import Thermometer
from datetime import timedelta
import gpflow
from copy import deepcopy
from Colormap import Colormap

class Data:
    """
    This class represents a dataset with associated metadata.
    """

    def __init__(self, dataframe, name, timeRange=None, no_scanning=None):
        """
        Initializes a Data object.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the dataset.
        - name (str): Name of the dataset.
        - timeRange (tuple): Time range of the dataset (default is None).
        - no_scanning: Information about scanning (default is None).
        """

        self.dataframe = dataframe
        self.name = name
        self.timeRange = timeRange
        self.no_scanning = no_scanning



class XperimentPlotter:
    """
    This class provides static methods for plotting and analyzing experimental data.
    """


    @staticmethod
    def ShortStats(data):
        """
        Prints short statistics about the provided data.

        Parameters:
        - data (list): List of Data objects.
        """

        print(f'***************')
        print(f'* STATISTICS: *')
        print(f'***************')
        
        print(f'Number of datasets: {len(data)}')
        names = []
        for d in data:
            names.append(d.name)
        print(f'Names of datasets: {names}')

        if ('temps' in data[0].dataframe):
            temp_merged = True
        else:
            temp_merged = False
        print(f'Temps merged: {temp_merged}')

        num_datapoints_mean = 0
        # num_datapoints_all = 0
        for d in data:
            num_datapoints_mean += len(d.dataframe['mean'])
        print(f'Number of datapoints (mean): {num_datapoints_mean:,}')


    @staticmethod
    def ReadData(partPath, title, timerange=None, no_scanning=None):
        """
        Reads data from a file in the specified path and returns a Data object.

        Parameters:
        - partPath (str): Path to the file containing the data.
        - title (str): Title of the Data object.
        - timerange (tuple): Time range of the dataset (default is None).
        - no_scanning: Information about scanning (default is None).

        Returns:
        Data: Data object containing the read data.
        """

        for file in os.listdir(partPath):
            if fnmatch.fnmatch(file, 'allMeasurments20*.csv'):
                return Data(pd.read_csv(f'{partPath}{file}', sep=','), title, timerange, no_scanning)


    @staticmethod
    def mergeTempsInData(d, path_temps):
        """
        Merges temperature data into the provided Data object.

        Parameters:
        - d (Data): Data object to merge temperature data into.
        - path_temps (str): Path to temperature data.

        Returns:
        Data: Updated Data object with merged temperature data.
        """

        startTimestamp = pd.to_datetime(d.timeRange[0], format='%Y-%m-%d %H:%M')
        endTimestamp = pd.to_datetime(d.timeRange[1], format='%Y-%m-%d %H:%M')
        timestamps = []
        num_of_col = len(d.dataframe.columns)
        for i in range(0, len(d.dataframe['levels'])):
                if num_of_col == 1001:
                        timedelay = 4.5
                else: # elif num_of_col == 655..
                        timedelay = 262
                timestamps.append(startTimestamp + timedelta(0, i * timedelay))
        if 'mean' not in d.dataframe:
            d.dataframe = pd.DataFrame({'levels': d.dataframe['levels'], 'mean': (d.dataframe.loc[:, d.dataframe.columns != 'levels']).mean(axis=1), 'timestamps': timestamps})
        else:
            d.dataframe = pd.DataFrame({'levels': d.dataframe['levels'], 'mean': d.dataframe['mean'], 'timestamps': timestamps})

        path_room = f'{path_temps}/localtemperature.log'
        path_monitor = f'{path_temps}/viewpixxII-temperature.log'
        room_t = Thermometer.ReadRoomTemperature(path_room, d.timeRange)
        monitor_t = Thermometer.ReadViewpixxTemperature(path_monitor, d.timeRange)
        room_t = room_t[room_t.temperature < 85.0]

        temps = pd.concat([room_t, monitor_t]).sort_values(by='datetime')
        # a little bit dirty, but allows an estimate
        temps = temps.interpolate()
        temps = pd.DataFrame({'datetime': temps['datetime'], 'temps': (temps.loc[:, ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7']]).mean(axis=1)})
        
        i=0
        if not temps.empty:
                for idx, row in temps.iterrows():
                        while i < len(d.dataframe) and row.datetime >= d.dataframe.iloc[i].timestamps:
                                d.dataframe.loc[i, 'temps'] = row.temps
                                i += 1
                        lasttemp = row.temps

                d.dataframe.fillna(lasttemp, inplace=True)
        
        return d


    @staticmethod
    def fitApproach(df, numberdatapoints, cuts, kernel, mean_funct=None, likelihood=None, iterationlimit=300, method='L-BFGS-B', showStd=False, path='tmp', baseFigurePath='./tmp/figures'):
        """
        Fits a Gaussian Process regression model to the provided data and plots the results.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - numberdatapoints (int): Number of data points to use for training.
        - cuts (tuple): Tuple specifying the range of data to consider.
        - kernel: GPflow kernel for the Gaussian Process.
        - mean_funct: Mean function for the Gaussian Process (default is None).
        - likelihood: Likelihood function for the Gaussian Process (default is None).
        - iterationlimit (int): Maximum number of optimization iterations (default is 300).
        - method (str): Optimization method for training the model (default is 'L-BFGS-B').
        - showStd (bool): Whether to display standard deviation in the plots (default is False).
        - path (str): Path for saving temporary files (default is 'tmp').
        - baseFigurePath (str): Base path for saving figures (default is './tmp/figures').

        Returns:
        gpflow.models.GPR: Trained Gaussian Process regression model.
        """
        # sort df
        df = df.sort_values(by='L_in')
        # cut df
        df = df[ df['L_in'] >= cuts[0]]
        df = df[df['L_in'] <= cuts[1]]
        print(f' slice: {cuts}')

        # extract data
        if numberdatapoints < len(df['L_in']):
            indices = np.linspace(0, len(df['L_in'])-1, numberdatapoints, dtype=int)
            X_train = np.array(df.iloc[indices][['L_in', 'temps']])
            Y_train = np.array(df.iloc[indices][['L_out']])
            print(f' num. training points: {len(indices)}')
        else:
            X_train = np.array(df[['L_in', 'temps']])
            Y_train = np.array(df[['L_out']])
            print(f' num. training points: {len(df["L_in"])}')



        # create and train model
        if not likelihood:
            model = gpflow.models.GPR((X_train, Y_train), kernel=deepcopy(kernel), mean_function=mean_funct, noise_variance=1e-3)
        else: 
            model = gpflow.models.GPR((X_train, Y_train), kernel=deepcopy(kernel), mean_function=mean_funct, likelihood=likelihood)
        opt = gpflow.optimizers.Scipy()
        summary = opt.minimize(model.training_loss, model.trainable_variables, method=method, options={'maxiter': iterationlimit})
        print(summary)
        
        gpflow.utilities.print_summary(model, fmt='notebook')
        try:
            print(model.mean_function.variables)
        except:
            print("no w")

        # 4. Compare
        mean_predictions, var_predictions = model.predict_y(np.array(df[['L_in', 'temps']]))
        std_predictions = np.sqrt(var_predictions)
        mean_predictions_n = np.asarray(mean_predictions, dtype=float)
        std_predictions_n = np.asarray(std_predictions, dtype=float)

        # first graph
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5), layout='constrained', subplot_kw={'projection': '3d'})
        fontsize = 18
        print(f'prediction by GP (max std: {std_predictions_n[:,0].max():.3f})')
        #train
        axes.scatter(X_train[:,0], X_train[:,1], Y_train, s=10, color=Colormap.black)
        axes.set_xlabel(r'$L_{in}$', fontsize=fontsize)
        axes.set_ylabel(r'temp [°C]', fontsize=fontsize)
        axes.set_zlabel(r'$L_{out} \left[\frac{cd}{m^2}\right]$', fontsize=fontsize)
        axes.view_init(elev=35, azim=-120);

        figurePath = f'{baseFigurePath}/{path}_A.png'
        fig.show()
        fig.savefig(figurePath, dpi=300)

        # second graph
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5), layout='constrained')
        fontsize = 18
        pcm = axes.scatter(df['L_in'], df['L_out']-mean_predictions_n[:,0], s=1, c=df['temps'], cmap='coolwarm', label=f'mean of residuen: {(df["L_out"]-mean_predictions_n[:,0]).mean():.3}\nstd. of residuen: {(df["L_out"]-mean_predictions_n[:,0]).std():.3}')
        print(f'mean of residuen: {(df["L_out"]-mean_predictions_n[:,0]).mean():.3}\nstd. of residuen: {(df["L_out"]-mean_predictions_n[:,0]).std():.3}')
        axes.fill_between(
                df['L_in'],
                - 1.96 * std_predictions_n[:,0],    # 95% prediction interval ~2 sigma
                + 1.96 * std_predictions_n[:,0],
                color=Colormap.map[1],
                alpha=0.2,
            )
        
        axes.set_xlabel(r'$L_{in}$', fontsize=fontsize)
        axes.set_ylabel(r'$L_{out}$ $\left[\frac{cd}{m^2}\right]$', fontsize=fontsize)
        ax2 = axes.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(r'temperature $\left[^{\circ}C\right]$', fontsize=fontsize)
        fig.colorbar(pcm)
        
        figurePath = f'{baseFigurePath}/{path}_B.png'
        fig.show()
        fig.savefig(figurePath, dpi=300)
        return model
    
    def predictApproach(df, cuts, model, showStd=False, path='tmp', baseFigurePath='./tmp/figures'):
        """
        Predicts outcomes using a pre-trained Gaussian Process regression model and plots the results.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - cuts (tuple): Tuple specifying the range of data to consider.
        - model (gpflow.models.GPR): Trained Gaussian Process regression model.
        - showStd (bool): Whether to display standard deviation in the plots (default is False).
        - path (str): Path for saving temporary files (default is 'tmp').
        - baseFigurePath (str): Base path for saving figures (default is './tmp/figures').
        """
        # sort df
        df = df.sort_values(by='L_in')
        # cut df
        df = df[ df['L_in'] >= cuts[0]]
        df = df[df['L_in'] <= cuts[1]]
        print(f' slice: {cuts}')

        # 4. Compare
        mean_predictions, var_predictions = model.predict_y(np.array(df[['L_in', 'temps']]))
        std_predictions = np.sqrt(var_predictions)
        mean_predictions_n = np.asarray(mean_predictions, dtype=float)
        std_predictions_n = np.asarray(std_predictions, dtype=float)

        # first graph
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5), layout='constrained', subplot_kw={'projection': '3d'})
        fontsize = 18
        # test
        axes.scatter(df[['L_in']], df[['temps']], df[['L_out']], s=1, alpha=0.5, color=Colormap.map[0], label='artificial data points')
        # prediction
        axes.scatter(df[['L_in']], df[['temps']], mean_predictions[:,0], s=1, label=f' prediction by GP (max std: {std_predictions_n[:,0].max():.3f})', color=Colormap.map[1])
        print(f'prediction by GP (max std: {std_predictions_n[:,0].max():.3f})')
        axes.set_xlabel(r'$L_{in}$', fontsize=fontsize)
        axes.set_ylabel(r'temp [°C]', fontsize=fontsize)
        axes.set_zlabel(r'$L_{out} \left[\frac{cd}{m^2}\right]$', fontsize=fontsize)
        axes.view_init(elev=35, azim=-120);

        figurePath = f'{baseFigurePath}/{path}_A.png'
        fig.show()
        fig.savefig(figurePath, dpi=300)

        # second graph
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5), layout='constrained')
        fontsize = 18
        pcm = axes.scatter(df['L_in'], df['L_out']-mean_predictions_n[:,0], s=1, c=df['temps'], cmap='coolwarm', label=f'mean of residuen: {(df["L_out"]-mean_predictions_n[:,0]).mean():.3}\nstd. of residuen: {(df["L_out"]-mean_predictions_n[:,0]).std():.3}')
        print(f'mean of residuen: {(df["L_out"]-mean_predictions_n[:,0]).mean():.3}\nstd. of residuen: {(df["L_out"]-mean_predictions_n[:,0]).std():.3}')
        axes.fill_between(
                df['L_in'],
                - 1.96 * std_predictions_n[:,0],    # 95% prediction interval ~2 sigma
                + 1.96 * std_predictions_n[:,0],
                color=Colormap.map[1],
                alpha=0.2,
            )

        axes.set_xlabel(r'$L_{in}$', fontsize=fontsize)
        axes.set_ylabel(r'$L_{out}$ $\left[\frac{cd}{m^2}\right]$', fontsize=fontsize)
        # set label for colorbar
        ax2 = axes.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(r'temperature $\left[^{\circ}C\right]$', fontsize=fontsize)
        fig.colorbar(pcm)
        
        figurePath = f'{baseFigurePath}/{path}_B.png'
        fig.show()
        fig.savefig(figurePath, dpi=300)