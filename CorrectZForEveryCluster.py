from Regions import *
from  gwyfile import load as gwyload
from  gwyfile.util import get_datafields , find_datafields
import pickle
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from matplotlib import path ### just for "drawing" an polygon for later extraction of the values inside this polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import multiprocessing 
from sklearn.neighbors import KernelDensity
import time
import warnings
from skimage import draw
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
from numpy.ma import masked_outside
import traceback
import logging
import beepy
from os import listdir



class clusterpic():
    """
    Creats an Object for working on STM data. 
    
    Class Instances:
        path: Path to the file
        name: name of file given by user
        data: 2d numpy array
        xres: number of points in x direction
        yres: number of points in y direction
        xreal: length of picture in x diektion  (e.g. 100nm)
        yreal: length of picture in y diektion 
        si_unit_xy: unitis of xreal/yreal e.g m or nm
        si_unit_z: unitis of z e.g m or nm
 
    """
    #import numpy as np
    def __init__(self, path = '', name ='', data = None , xres = None, yres= None ,
                 xreal = None, yreal= None, si_unit_xy= None, si_unit_z= None, metaData = None):
        self.path = path
        self.name = name
        self.data = data 
        self.xres = xres 
        self.yres = yres 
        self.xreal = xreal 
        self.yreal = yreal
        self.metaData = metaData
        self.currentSetPoint = None
        self.gapVoltage = None
        try:
            self.si_unit_xy = si_unit_xy.unitstr
        except:
            self.si_unit_xy = si_unit_xy
        try:
            self.si_unit_z = si_unit_z.unitstr
        except:
            self.si_unit_z = si_unit_z
            
        if (self.xreal is not None) and (self.yreal is not None):
            self.area = self.xreal * self.yreal
        else:
            self.area = None
        self.peak_XYdata = pd.DataFrame()
        self.clusters_coord = np.empty(0)
        self.ax = None
        self.pickable_artists = None
        self.event =None
        self.coor_regieons = []
        self.regions = []
        # self.tmp = [] # for debuging
        self.creat_heights_table()
        #self.heights = pd.DataFrame([])
        self.cluster_distribution = None
        self.nearest_neighbors_ditribution = None
        self.slope_map = []
        self.path_profiles ={} # for all profiles on the imshow() between clsuters cuts of the data so to say
        if (self.data.shape[0] == self.data.shape[1]) == False:
            #### correct cuted images to full dimetions, so it is simple to compute all other stuff
            xmeter_per_pix = self.xreal/self.xres
            ymeter_per_pix = self.yreal/self.yres
            dataFr = pd.DataFrame(self.data)
            self.data = dataFr.reindex(index=list(dataFr.columns)).to_numpy()
            self.yres = self.data.shape[0]
            self.xres = self.data.shape[1]
            self.yreal = ymeter_per_pix*self.data.shape[0]
            self.xreal = xmeter_per_pix*self.data.shape[1]
            
    def __repr__(self):
        return f"{self.name}"
    
    def creat_heights_table(self):
        self.heights = pd.DataFrame(columns = ['x',
                                               'y',
                                               f'x_{self.si_unit_xy}',
                                               f'y_{self.si_unit_xy}',
                                               'initial_Z', 
                                               'corrected_Z_averaged', 
                                               'corrected_Z_closest_step',
                                              'corrected_Z_highest_step'])
       
    def dump_picture(self,
                     prefix : str = None, 
                     sufix : str = '_clusterpic_obj',
                           folder : str = None) -> pickle:
        """
        Saves the hole clusterpic object into a pickel file with prefix and folder 
        """
        if prefix:
            dump_path = '[%s]%s%s.pkl' %( str(prefix) , self.name, str(sufix))
        else:
            dump_path = '%s%s.pkl' %(self.name, str(sufix))
        if folder:
            dump_path = folder+dump_path
        with open(dump_path, 'wb') as pickle_file:
            pickle.dump(self,pickle_file)

    def dump_calculated_heights(self, 
                           prefix = None, 
                           folder=None):
        if self.heights.empty:
            print('You try to dump empty pandas.DataFrame \n So no data for cluster heights were found. Run first finde_peaks_in_row(), group_clusters() or load cluster coordinates from pickle by runing load_dumped_clusters() and then calc_true_height_4_every_region()')
        else:
            if prefix:
                dump_path = '[%s]%s_cluster_heights.pkl' %( str(prefix) , self.name)
            else:
                dump_path = '%s_cluster_heights.pkl' %(self.name)
            if folder:
                dump_path = folder+dump_path
            with open(dump_path, 'wb') as pickle_file:
                pickle.dump(self.heights,pickle_file)
                
    def load_dumped_cluster_heights(self, path_to_pickle):
        """
        Loads cluster heights witch were dumeted by dump_cluster_heights()
        
        Parameters:
        -----------
            path_to_pickle: str 
                path to .pkl file
        """
        with open(path_to_pickle, "rb") as input_file:
            self.heights = pickle.load(input_file)

    def dump_cluster_coord(self, 
                           prefix = None, 
                           folder=None):
        if self.clusters_coord.size == 0:
            print('You try to dump empty array \n So no coordinates for cluster were found. Run first finde_peaks_in_row(), group_clusters() or load cluster coordinates from pickle by runing load_dumped_clusters()')
        else:
            if prefix:
                dump_path = '[%s]%s_cluster_coordinates.pkl' %( str(prefix) , self.name)
            else:
                dump_path = '%s_cluster_coordinates.pkl' %(self.name)
            if folder:
                dump_path = folder+dump_path
            with open(dump_path, 'wb') as pickle_file:
                pickle.dump(self.clusters_coord,pickle_file)

    def load_dumped_cluster_coord(self, path_to_pickle):
        """
        Loads cluster coordinates witch were dumeted by dump_cluster_coord()
        
        Parameters:
        -----------
            path_to_pickle: str 
                path to .pkl file
        """
        with open(path_to_pickle, "rb") as input_file:
            self.clusters_coord = pickle.load(input_file)

    
    def walk_to_the_extrema(self,test_data, xyz_current_max, extrema = 'max', pixelRange = 2):
        """
        Finds locle maxima by slicing test_data (NXN array) here STM image data with a window of +- PixelRange in first and second dimension. 
        It searches for local maxima in the slices and if there is no other maxima the search is aborted.

        Parameters:
        extrema: str: 'max' or 'min' if min find minima if max find maxima :)
        test_data: NxN numpy array, STM image data
        xyz_current_max: list of [x,y,z] coordinates of maximum/point from witch start searching
        pixelRange: integer, how big is the wind in wich to search. E.g pixelRange=4 produces 8X8 window (array with the shape = (8,8))
        """
        no_new_extrema_found = True
        suspect = xyz_current_max#[1],xyz_current_max[0], xyz_current_max[2]
        step_counter = 0
        while no_new_extrema_found:
            step_counter +=1
            y_range = [suspect[0]-pixelRange, suspect[0]+pixelRange]
            if y_range[0] < 0:y_range[0] = 0 # if you hit the boundaries important for correction later see maxXX
            if y_range[1] > test_data.shape[0] : y_range[1] = test_data.shape[0] # if you hit the boundaries 
            x_range = [suspect[1]-pixelRange, suspect[1]+pixelRange] # if you hit the boundaries
            if x_range[0] < 0 : x_range[0] = 0  # if you hit the boundaries
            if x_range[1] > test_data.shape[1] : x_range[1] = test_data.shape[1] # if you hit the boundaries
            aslice = test_data[x_range[0]:x_range[1],y_range[0]:y_range[1]]
            if extrema == 'min':
                extremaX, extremaY = np.unravel_index(aslice.argmin(), aslice.shape) ## finde maximum ids in 2d array slice
            else:
                extremaX, extremaY = np.unravel_index(aslice.argmax(), aslice.shape) ## finde maximum ids in 2d array slice
            extremaXX, extremaYY = extremaX+x_range[0], extremaY+y_range[0] ## correct for the actual array, so not the slice
            #self.tmp.append([aslice.max() ,suspect])
            if np.isnan(suspect[2]): ### some times it is just nan, no idea why. This is not very good fix but it works
                suspect[2] = 0.0
            if extrema == 'min':
                if aslice.min() < suspect[2]:
                    suspect = [extremaYY, extremaXX, aslice.min() ]
                
                    #no_new_max_found = False
                else:
                    no_new_extrema_found = False
            else:
                if aslice.max() > suspect[2]:
                    suspect = [extremaYY, extremaXX, aslice.max() ]
                
                    #no_new_max_found = False
                else:
                    no_new_extrema_found = False

        return suspect,y_range,x_range, aslice, step_counter
    
    def find_peaks_in_rows(self, 
                           prominence = 0.6E-09,
                           distance = 1,
                           height = 1.1E-09,
                           axes= 'both'):
        """
        Walk through rows or columns (or both) in 2d numpy array (self.data) and findes all peaks by aplaying pind_peaks from scipy

        Parameter:
            prominence: float, see scipy.signal.find_peaks for explanation
            distance: float, see scipy.signal.find_peaks for explanation
            height: float, see scipy.signal.find_peaks for explanation
            axes: str, 'rows', 'colums' or 'both' calculate peaks along row axes, column axes or both respectevly
        Reruns:
            data_frme with same dimenstion as test_data wiht peaks hiths and peaks positions 
            and nan values at other positions 
        """
        max_dic_columns = {}
        df_rows = None
        df_column = None
        if axes == 'rows' or axes == 'both':
            counter = 0
            for i in self.data: # iterate row by row
                xpeaks = find_peaks(i,height=height, distance=distance, prominence=(prominence,))[0]
                zpeaks = i[xpeaks] 
                max_dic_columns[counter] = {x:z for x,z in zip(xpeaks,zpeaks)}
                counter +=1 
            dataFr = pd.DataFrame(max_dic_columns).T
            dataFr.columns.name = "Y-Coordinate"
            dataFr.index.name = "X-Coordinate"
            dataFr = dataFr.reindex(columns=list(dataFr.index)) ### Fill all not existing rows with NaNs in order to get quadratik matrix again
            df_rows = dataFr.reindex(sorted(dataFr.columns), axis=1)

        if axes == 'columns' or axes == 'both':
            counter = 0
            for i in self.data.T: # iterate column by column
                xpeaks = find_peaks(i,  height=height, distance=distance, prominence=(prominence,))[0]
                zpeaks = i[xpeaks] 
                max_dic_columns[counter] = {x:z for x,z in zip(xpeaks,zpeaks)}
                counter +=1 
            dataFr = pd.DataFrame(max_dic_columns).T
            dataFr.columns.name = "X-Coordinate"
            dataFr.index.name = "Y-Coordinate"
            dataFr =  dataFr.reindex(columns=list(dataFr.index))### Fill all not existing rows with NaNs in order to get quadratik matrix again
            df_column = dataFr.reindex(sorted(dataFr.columns), axis=1).T
        if axes == 'rows':
            peak_data = df_rows
        elif axes == 'columns':
            peak_data = df_column
        elif axes == 'both':
            peak_data = df_column.combine_first(df_rows)

        self.peak_XYdata = peak_data

    def show_data(self,
                  ax =None,
                  cmap = 'gray',
                  data_multiplayer = 1,
                  cbar_on = True,
                  cbar_location = 'right',
                  cbar_fraction = 0.04740,
                  cbar_pad =  0.004,
                  bar = True,
                  bar_space_left = 0.05, # space in % from hole range
                  bar_space_bottom = 0.95,# space in % from hole range
                  bar_length = None, # if non will be calculatet automaticaly 10% of picture width
                  bar_color = 'white',
                  bar_size = 10,
                  bar_label_xshift = 1.5,   # in percent from origin
                  bar_label_yshift = 0.99,  # in percent from origin
                  bar_ticks = False,
                  mask = None,
                  unit = 'nm',
                  no_ticks =False,
                  show_clusters = False,
                  clusters_markersize = 3,
                  clusters_markercolor = 'r',
                  font_size= 10,
                  show_cluster_numbers = False,
                  cl_numb_fontsize = 8
                  ):
        """
    Plots the scanning tunneling microscope (STM) data.

    Parameters:
    -----------
    cmap: str
        The colormap used for the plot. Default is 'gray'.
    
    mask: list of lists:
        Plots only regions inside the numbers e.g. a,b for mask =[[a,b], [a1,b1], ...] 
    bar: bool
        Whether to show a bar on the plot or not. Default is True.
    bar_space_left: float
        The space from the left side of the plot to the bar, as a percentage of the x range. Default is 0.05.
    bar_space_bottom: float
        The space from the bottom of the plot to the bar, as a percentage of the y range. Default is 0.05.
    bar_length: float
        The length of the bar in nanometers. Default is 100.
    bar_color: str
        The color of the bar. Default is 'white'.
    bar_size: float
        The width of the bar in points. Default is 10.
    unit: str
        The unit for the axis labels. Default is 'nm'. Available are '$\mu$m' and $\AA$ for 10^-6 and 10^-10 m 
    no_ticks: bool
        Whether to show axis ticks or not. Default is False.

    Returns:
    --------
    fig, ax: tuple
        The figure and axis objects of the plot.
    """
        from matplotlib import ticker as mpl_ticker
        if not ax:
            fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': font_size})
        multiplayer = 1
        if not bar_length:
            match unit:
                case '$\AA$':
                    multiplayer = 1e10
                case 'nm':
                    multiplayer = 1e9
                case '$\mu$m':
                    multiplayer = 1e6
                    
            bar_length = round(self.xreal*0.1*multiplayer)
        if mask:
            for mski in mask:
                in_your_face =  masked_outside(self.data*data_multiplayer,mski[0],mski[1])
                im = ax.imshow(in_your_face,
                               cmap=cmap,
                                interpolation = None,
                               extent =[0, self.xreal*data_multiplayer, self.yreal*data_multiplayer, 0]
                               )
        else:
            im = ax.imshow(self.data*data_multiplayer,
                           cmap = cmap,
                           #origin = 'lower',
                           interpolation = None,
                           extent =[0, self.xreal*data_multiplayer, self.yreal*data_multiplayer, 0]
                           )
        if show_clusters:
            for i in range(0,len(self.clusters_coord)):
                ax.plot(self.clusters_coord[:,0][i]*(self.xreal/self.xres)*data_multiplayer,
                    self.clusters_coord[:,1][i]*(self.yreal/self.yres)*data_multiplayer,
                        'o', c = clusters_markercolor, ms = clusters_markersize)
            
        if bar:
            ax.hlines(self.yreal*bar_space_bottom*data_multiplayer,
                       #1E-8,
                       xmin= self.xreal*bar_space_left*data_multiplayer,
                       xmax = self.xreal*bar_space_left*data_multiplayer + bar_length*1e-9*data_multiplayer,
                       colors = bar_color,
                       linewidth = bar_size)
    
            ax.annotate(str(bar_length)+' '+unit,
                    (self.xreal*bar_space_left*bar_label_xshift*data_multiplayer,
                     self.yreal*bar_space_bottom*bar_label_yshift*data_multiplayer),
                             
                             color = bar_color)
        
        if unit == 'nm':
            func = lambda x,pos: "{:g}".format(x*1e9)
        if unit == '$\mu$m':
            func = lambda x,pos: "{:g}".format(x*1e6)
        if unit == '$\AA$':
            func = lambda x,pos: "{:g}".format(x*1e10)
        
        
        fmt = mpl_ticker.FuncFormatter(func)

        if no_ticks:
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax.set_xticks([])
            ax.set_yticks([])
        if cbar_on:
            cbar = plt.colorbar(mappable= im, fraction=cbar_fraction, pad=cbar_pad, location = cbar_location)
            tick_locator = ticker.MaxNLocator(nbins=9)
            cbar.locator = tick_locator
            cbar.ax.tick_params(direction = 'out')
            cbar.update_ticks()
            cbar.ax.set_xticks(cbar.ax.get_xticks()) ## strange avoding of error
            cbar.ax.set_yticks(cbar.ax.get_yticks())
            ticklabs = cbar.ax.get_yticklabels()
            cbar.ax.set_yticklabels(ticklabs, fontsize=3)
            cbar.ax.set_title(r'\si{\nano\meter}', fontsize = 1, pad = 1)
        if show_cluster_numbers:
            for i in range(0,len(self.clusters_coord)):
                ax.annotate(i, (self.clusters_coord[:,0][i]*(self.xreal/self.xres)*data_multiplayer, self.clusters_coord[:,1][i]*(self.yreal/self.yres)*data_multiplayer), fontsize=cl_numb_fontsize)
            
        return ax
    
    def intensity_profile_along_path(self, counter = 1, cluster_numbers = 'all', data_multiplayer = 1):
        """
        Generate intensity profile along a specified path.

        Parameters:
        - cluster_numbers (str or list): If 'all', consider all clusters. If a list, specify cluster indices.
                                         If a list with two elements, generate a straight line between two clusters.
                                         If a list with more than two elements, generate a zigzag line between specified clusters.

        - data_multiplier (float): Multiplier to scale the intensity values.

        Returns:
        numpy.ndarray: Array containing information along the path. Each row represents a point on the path with columns:
                       [row_index, column_index, distance_from_start, intensity_value].

        Example:
        intensity_profile_along_path(cluster_numbers=[0, 1, 3], data_multiplier=1e9) # 1e9 for representing  nm
        """
        rr, cc = [], []
        if cluster_numbers == 'all':
            points = self.clusters_coord[:,:2].astype(int)
        elif isinstance(cluster_numbers,list):
            if len(cluster_numbers) == 2: # strait line between two clusters
                points = (self.clusters_coord[cluster_numbers[0]][:2].astype(int) ,
                          self.clusters_coord[cluster_numbers[1]][:2].astype(int)
                          )
            else: ## zig zag line between all clusters in the list cluster_numbers
                points = self.clusters_coord[cluster_numbers][:,:2].astype(int)
        else:
            warnings.warn(f"This is not a list {cluster_numbers}") 
        # Generate indices along the path for each segment
        for i in range(len(points) - 1):
            segment_rr, segment_cc = draw.line(*points[i], *points[i + 1])
            rr.extend(segment_rr)
            cc.extend(segment_cc)

        # Clip indices to stay within image boundaries
        rr = np.clip(rr, 0, self.data.shape[0] - 1)
        cc = np.clip(cc, 0, self.data.shape[1] - 1)

        # Extract intensity values along the path
        intensity_profile = self.data[cc,rr]
        dd_zero = [rr[0],cc[0]]#np.sqrt((rr[0]*(self.xreal/self.xres))**2. + cc[0]*(self.yreal/self.yres)**2.)# [rr[0],cc[0]]
        dd = []
        dsum = 0
        for x,y in zip(rr,cc):
            d = np.sqrt(((x-dd_zero[0])*(self.xreal/self.xres))**2. + ((y-dd_zero[1])*(self.yreal/self.yres))**2.)
            dsum = dsum + d
            dd.append(dsum)
            dd_zero = [x,y]
        dd = np.array(dd)
        profiles = np.array((rr,cc,dd*data_multiplayer,intensity_profile)).T
        self.path_profiles[counter] ={'x_pix':profiles[:,0],'y_pix':profiles[:,1],'x':profiles[:,0]*(self.xreal/self.xres)*data_multiplayer,'y':profiles[:,1]*(self.yreal/self.yres)*data_multiplayer,'distance':profiles[:,2], 'intensity':profiles[:,3]}
        return profiles
    
    def intensity_profile_along_path_multi(self, list_of_all_cluster_pairs =[], data_multiplayer = 1):
        """
        Generate intensity profiles along specified paths for multiple cluster pairs.

        Parameters:
        - list_of_all_cluster_pairs (list): List of cluster pairs, where each pair is represented as [cluster_index_1, cluster_index_2].
                                            The function will generate intensity profiles for each specified pair.

        - cluster_numbers (str or list): If 'all', consider all clusters. If a list, specify cluster indices.
                                         If a list with two elements, generate a straight line between two clusters.
                                         If a list with more than two elements, generate a zigzag line between specified clusters.

        - data_multiplier (float): Multiplier to scale the intensity values.

        Data is stored in self.path_profiles attribute
        Returns:
        list: List containing intensity profiles for each specified pair. Each element in the list is an array containing
              information along the path for a pair. Each row represents a point on the path with columns:
              [row_index, column_index, distance_from_start, intensity_value].

        Example:
        intensity_profile_along_path_multi(list_of_all_cluster_pairs=[[0, 1], [2, 3]], cluster_numbers=[0, 1, 3], data_multiplier=1e9)
        """
        result =[]
        counter = 0
        if len(list_of_all_cluster_pairs)==0:
            warnings.warn(f"Pleas provide the list of lists for extraction of paths between cluster pairs a1,a1 and a2,a2  like [[a1,a1], [a2,a2]]") 
        else:
            for i in list_of_all_cluster_pairs:
                 result.append(self.intensity_profile_along_path(counter = counter, cluster_numbers = i, data_multiplayer = data_multiplayer))
                 counter +=1
        return result
    
    def get_peakXYdata(self):
        """
        Calculates 2d numpy array of XY data of peaks estimatet by finde_peaks_in_rows().
            it is just the heigest points not the maxima of cluster, you need to run group_clusters() for this
        """
        return self.peak_XYdata
    
    def calculate_angle_betwee_3_clusters(self, cls1, cls2, cls3):
        """
        Calculates the angle in degrees between three clusters in a 2D space.

        Parameters:
        - cls1 (int): Index of the first cluster.
        - cls2 (int): Index of the second cluster (common vertex).
        - cls3 (int): Index of the third cluster.

        Returns:
        float: The angle in degrees between the vectors formed by the clusters.
        """
        vector1 = np.array(self.clusters_coord[cls1]) - np.array(self.clusters_coord[cls2])
        vector2 = np.array(self.clusters_coord[cls3]) - np.array(self.clusters_coord[cls2])

        # Calculate the cosine of the angle
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        # Use arccosine to get the angle in radians
        angle_radians = np.arccos(cosine_angle)

        # Convert radians to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    def distance_between_clusters(self, first, second):
        """
        calculates distance between clusters by number so distance between first and second cluster would be self.distance_between_cluster(1,2)
        """
        nm_per_pix = (self.xreal/self.xres)
        return np.sqrt( (self.clusters_coord[first][0]*nm_per_pix -  self.clusters_coord[second][0]*nm_per_pix)**2.
                         + (self.clusters_coord[first][1]*nm_per_pix -  self.clusters_coord[second][1]*nm_per_pix)**2.
                       )
    
    def finde_nn_in_r(self, dr = np.sqrt(1e-9**2+1e-9**2), count_solo_cluster = False):
        """
        Finds the nearest neighbors within a given distance for each coordinate in `coord`.

        Parameters:
        -----------
        dr: float, optional
            The distance threshold within which neighbors are considered. Default is the square root of 2 nanometers squared.
        
        Returns:
        --------
        result: dict
            A dictionary where the keys are the number of nearest neighbors and the values are the count of points having that number of neighbors within the given distance.
        """
        coord = self.get_xy_coord()
        tree = cKDTree(coord)
        l = []
        for i in coord:
            nn = tree.query_ball_point(i, r = dr)#, return_length = True)
            if nn:
                tmp_count = 0
                for k in nn: 
                    if (coord[k] == i).all(): #skip counting current cluster as neighbor
                        pass
                    else:
                        tmp_count +=1 
                #print(tmp_count)
                l.append(tmp_count)
        nn_compl = np.arange(1,max(l)+1)
        count = []
        for p in nn_compl:
            counter = 0
            for s in l:
                if s == p:
                    counter +=1 
            count.append(counter)
        #return(pd.DataFrame(np.vstack((nn_compl,np.array(count)))))
        #return(pd.DataFrame((nn_compl,np.array(count)))
        result = {}
        if count_solo_cluster:
            solo_clusters = len(coord) - np.array(count).sum()
            result[0] = solo_clusters
        for i,k in zip (nn_compl,count):
            result[i] = k
        return result

    def show_peakXYdata(self, figsize = (10,10) , cmap = None, returnImag = False):
        """
        Plots image of all peaks estimatet by find_peaks_in_rows() (plt.imshow())
        
        """
        if not self.peak_XYdata.empty:
            plt.figure(figsize=figsize)
            Peakimage = plt.imshow(self.peak_XYdata,interpolation='nearest', aspect='auto', cmap = cmap)
            if returnImag:
                return Peakimage
        else:
            print("No data found, run first find_peaks_in_rows()")
    
    def group_clusters(self, max_d = 30, delet_at_edge = False, xres = None, yres=None):
        """
        Group clusters with the data from finde_peaks_in_rows() by clustering the peak data due to the neighbours distance

        Parameters:
            peak_data: 2d numpy.array, peak data produced by finde_peaks_in_rows()
            max_d: int, max distance of contiguous clusters

        Returns:
            clusters_coord: Nx3 numpy.array, x,y,z data of all found clusters
        """
        coordinate_list = []
        max_values_of_clusters = []## same shape as coordinate_list
        clusterizerMe = self.peak_XYdata.to_numpy().T
        for ix, iy in np.ndindex(clusterizerMe.shape):
            if not np.isnan(clusterizerMe[ix][iy]):
                coordinate_list.append([ix,iy,clusterizerMe[ix][iy]])
        coordinate_list = np.array(coordinate_list)
        Z = linkage(coordinate_list,
                method='average',  # dissimilarity metric: max distance across all pairs of 
                                    # records between two clusters
                metric='euclidean'
        )                           # you can peek into the Z matrix to see how clusters are 
                                    # merged at each iteration of the algorithm

        clusters = fcluster(Z, max_d, criterion='distance')  
        all_clusters_list = [coordinate_list[np.where(clusters==k)[0].tolist()] for k in np.unique(clusters)]
        clusters_coord =np.array([i[np.where(i[:,2]==i[:,2].max())][0] for i in all_clusters_list])### finde row with maximum value in 3erd columns
        if delet_at_edge:
            list_for_deletion = []
            index_for_deletion = -1
            for i in clusters_coord:
                index_for_deletion += 1
                if i[0] == 0 or i[0] == xres or i[1] == 0 or i[1] == yres:
                        list_for_deletion.append(index_for_deletion)
            if index_for_deletion != -1:                
                clusters_coord = np.delete(clusters_coord,list_for_deletion,axis=0)
        self.clusters_coord = clusters_coord
        
        
    def update_peaked_clusters(self, pickable_artists, xyz =None, max_crawler = False, extrema = 'max'):
        """
        Updates the list of clusters withch was peakt by cluster_peaker() or by given list xyz
        Parameters:
            data: 2d numpy array, STM Image
            pickable_artists: matplotlib.pickable_artists
            xyz: [x,y,z] or list of [[x1,y1,z1], [x2,y2,z2], ...] coordinates if something gets wrong with the UI peaking cluster
            max_crawler: If True an xy is given, uses walk_to_the_extrema() for finding maximum/minimum
        Returns:
            clusters_coord: nX3 numpy array ([x1,y1,z1],[x2,y2,z2], ... )
        """
        clusters_coord = np.array([[i.get_data()[0][0],i.get_data()[1][0],
                                  self.data[int(i.get_data()[1][0])][int(i.get_data()[0][0])]] for i in pickable_artists])
        if xyz:
            if any(isinstance(el, list) for el in xyz): ### only if it is list of lists
                new_xyz = []
                for i in xyz:
                    if max_crawler:
                        new_xyz.append(np.array(self.walk_to_the_extrema(self.data,i, extrema = extrema)[0]))
                    else:
                        new_xyz.append(i)
                clusters_coord = np.vstack((clusters_coord,np.array(new_xyz)))
            else:
                if max_crawler:
                    new_xyz = self.walk_to_the_extrema(self.data,xyz, extrema = extrema)[0]
                    clusters_coord = np.vstack((clusters_coord,np.array(new_xyz)))
                else:
                    clusters_coord = np.vstack((clusters_coord,np.array([int(xyz[0]),int(xyz[1]),self.data[int(xyz[0])][int(xyz[1])]])))
        clusters_coord = np.unique(clusters_coord, axis = 0) ### erase duplicates
        self.clusters_coord = clusters_coord

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all([v >= 0 for v in vertices]):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)
    def get_xy_coord(self):
        return  np.array((self.clusters_coord[:,0]*(self.xreal/self.xres),
                              self.clusters_coord[:,1]*(self.yreal/self.yres))).T # prepare 2d array vor surching distance
    def calc_cluster_distribution(self):
        """
        Calculates all distances c = sqrt((a1-a2)**2 + (b1-b2)**2) from calculated cluster heigts table. So calc_true_height_4_every_region() have to be run befor
        All distances are stored in the class attribute
        'cluster_distribution'.
      
        Returns:
        --------
            None
        """
        # all_coord = np.array((self.heights['x']*(self.xreal/self.xres),
        #                       self.heights['y']*(self.yreal/self.yres))).T # prepare 2d array vor surching distance
                                               
        # all_coord = np.array((self.heights[f'x_{self.si_unit_xy}'],
        #                       self.heights[f'y_{self.si_unit_xy}'])).T # prepare 2d array vor surching distance

        all_coord = np.array((self.clusters_coord[:,0]*(self.xreal/self.xres),
                              self.clusters_coord[:,1]*(self.yreal/self.yres))).T # prepare 2d array vor surching distance

        distribution = []
        for i in range(0, len(all_coord)):
            for k in range(0, len(all_coord)):
                if k!=i:
                    distribution.append(np.sqrt( (all_coord[k][0] - all_coord[i][0])**2. 
                                            + ( all_coord[k][1]  - all_coord[i][1])**2. ))
        self.cluster_distribution = np.array(distribution)
        
    def calc_nn_distribution(self):
        """
        Calculates all distances c = sqrt((a1-a2)**2 + (b1-b2)**2) from calculated cluster heigts table. So calc_true_height_4_every_region() have to be run befor
        All distances are stored in the class attribute
        'nn_distribution'.
        Returns:
        --------
            None
        """
        # all_coord = np.array((self.heights['x']*(self.xreal/self.xres),
        #                       self.heights['y']*(self.yreal/self.yres))).T # prepare 2d array vor surching distance
                                               
        # all_coord = np.array((self.heights[f'x_{self.si_unit_xy}'],
        #                       self.heights[f'y_{self.si_unit_xy}'])).T # prepare 2d array vor surching distance

        all_coord = np.array((self.clusters_coord[:,0]*(self.xreal/self.xres),
                              self.clusters_coord[:,1]*(self.yreal/self.yres))).T # prepare 2d array vor surching distance

        distribution = []
        for i in range(0, len(all_coord)):
            tmp_nn_distribution = []
            for k in range(0, len(all_coord)):
                if k!=i:
                    tmp_nn_distribution.append(np.sqrt( (all_coord[k][0] - all_coord[i][0])**2. 
                                            + ( all_coord[k][1]  - all_coord[i][1])**2. ))
            distribution.append(min(tmp_nn_distribution))
        self.nn_distribution = np.array(distribution)
    @staticmethod
    def calculate_slope(array):
        rows, cols = array.shape
        slope_matrix = np.zeros_like(array, dtype=float)
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                z_pp = array[i+1, j+1]
                z_op = array[i+1, j]
                z_mp = array[i+1, j-1]
                z_po = array[i, j+1]
                z_mo = array[i, j-1]
                z_pm = array[i-1, j+1]
                z_om = array[i-1, j]
                z_mm = array[i-1, j-1]
                
                dz_dx = (( z_pp + 2. * z_po + z_pm ) - ( z_mp + 2 * z_mo + z_mm ))/8
                dz_dy = (( z_pp + 2. * z_op + z_mp ) - ( z_pm + 2 * z_om + z_mm ))/8
                
                slope = np.sqrt(dz_dx**2 + dz_dy**2)
                slope_matrix[i, j] = np.degrees(np.arctan(slope))

        slope_matrix = slope_matrix/slope_matrix.max() #normalize
        return slope_matrix
    
    def cut_image_regions(self, window = None):
        """
        Cuts the image data in to regions around clusters determine by Voronoi algorithm if window id None. If window is given: cuts image in squers with the length of window
        
        Args:
            window (int): length of a side of a rectangular for the window bilding arraound a cluster 
        """
        if not self.heights.empty:
            self.creat_heights_table()
        if np.all(self.slope_map):
            self.slope_map = clusterpic.calculate_slope(self.data)
        if window is not None:
            region_type = f'rectangular {window} pix'
            self.heights.index.name = region_type
            self.regions = []
            for idx,i in enumerate(self.clusters_coord): # sclice data in to reagion bei squers
                y_range = [int(i[1]-window), int(i[1]+window)]
                if y_range[0] < 0:y_range[0] = 0 # if you hit the boundaries important for correction later see maxXX
                if y_range[1] > self.data.shape[0] : y_range[1] = self.data.shape[0] # if you hit the boundaries 
  
                x_range = [int(i[0]-window), int(i[0]+window)]
                if x_range[0] < 0 : x_range[0] = 0  # if you hit the boundaries
                if x_range[1] > self.data.shape[1] : x_range[1] = self.data.shape[1] # if you hit the boundaries
  
                #print(x_range, y_range)
                aslice = self.data[y_range[0]:y_range[1],x_range[0]:x_range[1]]
                #maxX, maxY = np.unravel_index(aslice.argmax(), aslice.shape) ## finde maximum ids in 2d array slice
                #maxXX, maxYY = maxX+x_range[0], maxY+y_range[0] ## correct for the actual array, so not the slice
                #[maxYY, maxXX, aslice.max() ]
                slice_xyz = []
                for y in range(aslice.shape[0]):
                    for x in range(aslice.shape[1]):           
                        slice_xyz.append([x+x_range[0],
                                          y+y_range[0],
                                          aslice[y][x]])  
                #self.coor_regieons.update({idx:{'x_min_id_offset':x_range[0], 'y_min_id_offset':y_range[0],'slice':aslice,'xyz(max)':i}})
                #self.coor_regieons.append([np.array(slice_xyz),i])
                a_region = region()
                a_region.region_id = idx
                a_region.coordinates = np.array(slice_xyz)
                a_region.slope_map = self.slope_map[y_range[0]:y_range[1],x_range[0]:x_range[1]]
                a_region.slope_map = a_region.slope_map/a_region.slope_map.max() #normolize
                a_region.cluster_peak_coordinates = np.array(i)
                a_region.region_type = region_type
                self.regions.append(a_region)
        else:
            self.regions = []
            region_type = 'voronoi'
            self.heights.index.name = region_type
            vor = Voronoi(np.vstack((self.clusters_coord[:,0], self.clusters_coord[:,1])).T, qhull_options='Qbb Qc Qx')
            regions, vertices = self.voronoi_finite_polygons_2d(vor)### convert regions in to finit regions
            coor_regieons = []
            xyz_data = []
            xyz_slope_map = []
            for iy, ix in np.ndindex(self.data.shape): ###Reshaping NxN array int too 3XN .shoulb be vectorized!!!! 
                xyz_data.append([ix,iy,self.data[ix,iy]])
                xyz_slope_map.append([ix,iy,self.slope_map[ix,iy]]) 
            xyz_data = np.array(xyz_data)
            xy_data = np.array([xyz_data[:,1],xyz_data[:,0]]).T ### reduce from 3XN to 2XN needed for matplotlib.path.path.contains_points
            xyz_slope_map = np.array(xyz_slope_map)
            xy_slope_map = np.array([xyz_slope_map[:,1],xyz_slope_map[:,0]]).T ### reduce from 3XN to 2XN needed for matplotlib.path.path.contains_points
            for idx,i in enumerate(regions): ##### Very Very slow need vectorization!!!!!!
                polygon_coord = path.Path(vertices[i]) ### extracting coordinates out of polygon
                a2d_coordinates_of_area = polygon_coord.contains_points(xy_data,radius=0.0)
                a2d_coordinates_of_slope = polygon_coord.contains_points(xy_slope_map,radius=0.0)

                a3d_area = xyz_data[a2d_coordinates_of_area]
                a3d_slope_map = xyz_slope_map[a2d_coordinates_of_slope]
                dict_a3d_area = {}
                for i in a3d_area:
                     dict_a3d_area[(i[0],i[1])] = i[2]
                breakIt = False
                cluster_max_inthearea = None
                for k in self.clusters_coord:
                    if breakIt:
                        break
                    if (k[1],k[0]) in dict_a3d_area:
                        # cluster_max_inthearea =(i[0],i[1],dict_a3d_area[(i[0],i[1])])
                        cluster_max_inthearea =(k[1],k[0],k[2])
                        breakIt = True
                        break
                a_region = region()
                a_region.region_id = idx
                a_region.coordinates = a3d_area
                a_region.slope_map = a3d_slope_map
                a_region.slope_map[:,2] = a_region.slope_map[:,2]/a_region.slope_map[:,2].max() #normolize
                a_region.cluster_peak_coordinates = np.array(cluster_max_inthearea)
                a_region.region_type = region_type
                self.regions.append(a_region) 
                #self.coor_regieons.update({idx:{'x_min_id_offset':x_range[0], 'y_min_id_offset':y_range[0],'slice':aslice,'xyz(max)':cluster_max_inthearea}})
                
        
    
        
    def parallel_correct_height(self, 
                        slope_threshold_factor = 0.1, 
                         groundlevel_cutoff = 0.30, 
                         method='complete',
                         metric='minkowski',
                         threshold = 'default', 
                         thold_default_factor = 1.1,
                         cutoff_points = 5,
                         seek_for_steps = 'False'):
        
        """
        Calculates the heights in parallel for image regions.

        Args:
            slope_threshold_factor (float): Factor for slope threshold (default 0.1).
            groundlevel_cutoff (float): Ground level cutoff (default 0.30).
            method (str): Method for correction (default 'complete').
            metric (str): Metric for correction (default 'minkowski').
            threshold (str): Threshold for correction (default 'default' : Some sort of coordinates mean of ground level sea self.region() ) 
            thold_default_factor (float): Factor for default threshold (default 1.1).
            cutoff_points (int): Cutoff points for correction (default 5).
            seek_for_steps (bool): Flag indicating whether to seek for steps (default False).

        Raises:
            Warning: If no regions are found. Please use self.cut_image_regions() first.

        Returns:
            None

        Note:
            The function utilizes multiprocessing for parallel correction of heights for each region found in the image.
    """
       
        #if not self.heights.empty:
        #    self.creat_heights_table()
        if not self.regions:
            warnings.warn("No regions found. Please use self.cut_image_regions() first")
        for region in self.regions: #set some variables
            region.slope_threshold_factor = slope_threshold_factor 
            region.groundlevel_cutoff = groundlevel_cutoff
            region.method=method
            region.metric=metric
            region.threshold = threshold 
            region.thold_default_factor = thold_default_factor
            region.cutoff_points = cutoff_points
            region.seek_for_steps = seek_for_steps
        a_pool = multiprocessing.Pool()
        try:    
            self.regions = a_pool.map(self.correct_heights_4_regions,
                                      self.regions)
            a_pool.close()
        except Exception as e:
            beepy.beep(sound=3)
            a_pool.close()
            logging.error(traceback.format_exc())
        for idx,i in enumerate(self.regions):# collecting corrected hights
            self.update_height(i, idx)
            
    def correct_heights_4_regions(self, i):
        """
        just needed fo parallel_correct_height()
        """
        i.find_groundlevel()
        i.calc_true_hight()
        #print(f'{i.region_id} Done')
        return i

    def calc_true_height_4_every_region(self, 
                         break_index = None,
                         slope_threshold_factor = 0.1, 
                         groundlevel_cutoff = 0.30, 
                         method='complete',
                         metric='minkowski',
                         threshold = 'default', 
                         thold_default_factor = 1.1,
                         cutoff_points = 5,
                         seek_for_steps = 'False'):
        
        """
        Correct the heights in z for every reagion (witch were cuted by cut_image_regions)
        
        Parameters:
             slope_threshold_factor: float
                 e.g. 0.01 means 1% of max-min sloap value would be consiedert as posible ground level points 
                 default = 0.1 
             groundlevel_cutoff: float 
                 e.g. 0.3 means all values in ground level candidate points smaller than 30% of maximum of cluster would be consiedert as ground level
                 default = 0.30 
             method: str: 
                method used by scipy.cluster.hierarchy.linkage
                default = 'complete'
            metric: str:
                metric used by scipy.cluster.hierarchy.linkage
                default = 'minkowski'
            threshold: str or 
                metric used by scipy.cluster.hierarchy.linkage
                if default a squear root of (x_min - x_max)**2 + (y_min-y_max) times thold_default_factor
                where x_ , y_ are coordinates
            thold_default_factor: float
                scaling factor for threshold, this is arbitary and depends on quality and/or spacing  of data
                default = 1.1,
                
            cutoff_points: int 
                number of points in a ground_level clusterd points, witch would be considered as artifacte e.g. 'jumping' stm tip/nois
                default is 5. But you may set it iven to 0 (all points then includet)
                
            seek_for_steps: string 
                if 'False' correction of z value only by avareging all point in ground level
                if 'True' correction of z value only by avareging all point in nearst cluster of points in ground level
                if 'both' correct both
                defalut = 'False'
        Returns:
            pandas.DataFrame with 4 or 5 colums:
                if seek_for_steps = 'False' 
                    ['x','y','z', 'z-z_avarage(nearest_step)']
                if seek_for_steps = 'True' 
                    ['x','y','z', 'z-z_average(ground_level)']
                if seek_for_steps = 'both'
                    ['x','y','z', 'z-z_average(ground_level)', 'z-z_avarage(nearest_step)']   
        """
        
        for idx,i in enumerate(self.regions):
            i.slope_threshold_factor = slope_threshold_factor 
            i.groundlevel_cutoff = groundlevel_cutoff
            i.method=method
            i.metric=metric
            i.threshold = threshold 
            i.thold_default_factor = thold_default_factor
            i.cutoff_points = cutoff_points
            i.seek_for_steps = seek_for_steps
            i.find_groundlevel()
            i.calc_true_hight()
            self.update_height(i,idx)
            print(idx)
            if break_index:
                if idx == break_index:
                    break
            
        
    def cluster_peaker(self,
                       cluster_numbers = True, 
                       markersize = 3,
                       figsize=(10,10), 
                       pixelRange = 2,
                       pikerRange = 10,
                       mask =None,
                       data_multiplayer = 1,
                       cmap = 'grey',
                       show_regions = False,
                       face_color = 'red',
                       rim_color = 'black',
                       extrema = 'max',
                       title_fontsize = 8,
                       alpha = 0.3):
        """
            Plots cluster coordinates over data and allows deletion and addition of cluster positions by mouse click.

            Parameters:
            - cluster_numbers (bool): If True, cluster indices are annotated on the plot.
            - markersize (int): Size of the markers representing clusters.
            - figsize (tuple): Figure size (width, height) in inches.
            - pixelRange (int): Pixel range used for finding extrema when adding a new cluster.
            - pikerRange (int): Picker range for identifying a cluster marker during mouse click.
            - show_regions (bool): If True, regions such as Voronoi patterns or rectangles are shown on the plot.
            - face_color (str): Face color of the region patches.
            - rim_color (str): Rim color of the region patches.
            - extrema (str): Specifies whether to walk to the 'max' or 'min' extrema when adding a new cluster.
            - alpha (float): Alpha value for controlling the transparency of the region patches.

            Returns:
            - matplotlib.axes: The plot's axes.
            - pickable_artists (list): List of x, y, z cluster coordinates, with potential additions or deletions.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if show_regions:
            if self.regions:
                for i in self.regions:
                    if i.region_type == 'voronoi':
                        vor = Voronoi( np.vstack((self.clusters_coord[:,0], self.clusters_coord[:,1])).T, qhull_options='Qbb Qc Qx')
                        voronoi_plot_2d(vor, ax = ax, show_points = False)
                        break
                    elif 'rectangular' in i.region_type:
                        x_min , x_max, y_min, y_max =min(i.coordinates[:,0]), max(i.coordinates[:,0]), min(i.coordinates[:,1]), max(i.coordinates[:,1])
                        rectangle = plt.Rectangle((x_min,y_min), x_max - x_min, y_max - y_min, fc=face_color,ec=rim_color, alpha = alpha)
                        ax.add_patch(rectangle)
        #ax.imshow(self.data)
        if mask:
            for mski in mask:
                in_your_face =  masked_outside(self.data*data_multiplayer,mski[0],mski[1])
                im = ax.imshow(in_your_face,
                               cmap=cmap,
                                interpolation = None,
                               #extent =[0, self.xreal*data_multiplayer, self.yreal*data_multiplayer, 0]
                               )
        else:
            im = ax.imshow(self.data*data_multiplayer,
                           cmap = cmap,
                           #origin = 'lower',
                           interpolation = None,
                           #extent =[0, self.xreal*data_multiplayer, self.yreal*data_multiplayer, 0]
                           )
        
        pickable_artists = []
        for i in range(0,len(self.clusters_coord)):
            pt, = ax.plot(self.clusters_coord[:,0][i],self.clusters_coord[:,1][i], 'o', c = 'r', ms = markersize)  
            pickable_artists.append(pt)

        coords = []
        new_maxes =[]

        removable = []
    
        def onclick(event,pickable_artists,gwy_data, extrema = extrema):
            if event.inaxes is not None and not hasattr(event, 'already_picked'):
                ax = event.inaxes

                remove = [artist for artist in pickable_artists if artist.contains(event)[0]]

                if not remove:
                    # add a pt        
                    x, y = ax.transData.inverted().transform_point([event.x, event.y])
                    new_max = self.walk_to_the_extrema(self.data,[int(x),int(y),self.data[int(x)][int(y)]],pixelRange=pixelRange, extrema = extrema)
                    pt, = ax.plot(new_max[0][0], new_max[0][1], 'o', picker=pikerRange, c='r')
                    pickable_artists.append(pt)
                    removable.append(new_max)

                else:
                    removable.append(remove)
                    pickable_artists.remove(remove[-1])
                    for artist in remove:
                        artist.remove()
                plt.draw()
        self.cid = fig.canvas.mpl_connect('button_press_event',lambda event: onclick(event,pickable_artists,self.data, extrema = extrema))

        # ax.set_xlim(vor.min_bound[0] - 10, vor.max_bound[0] + 10)
        # ax.set_ylim(vor.min_bound[1] - 10, vor.max_bound[1] + 10)

        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])

        if cluster_numbers:
            for i in range(0,len(self.clusters_coord)):
                ax.annotate(i, (self.clusters_coord[:,0][i], self.clusters_coord[:,1][i]), fontsize=8)

        ax.set_title('Total clusters: '+str(len(self.clusters_coord)), fontsize = title_fontsize,)
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()                     # and move the X-Axis
        fig.set_facecolor('white')
        return ax, pickable_artists
    
    def update_height(self, region, index):
        y,x,z = region.cluster_peak_coordinates
        self.heights.loc[index] = [x,
                                   y,
                                   x*self.xreal/self.xres,
                                   y*self.yreal/self.yres,
                                   z,
                                   region.true_hight,
                                   region.true_hight_closest_ground_level,
                                   region.true_hight_heighest_ground_level]
    def height_distribution(self, bins = "auto"):
        """
        Plot the distribution of heights.

        Args:
            bins (str or dict): Specification for the number of bins in the histogram.
                - If 'auto', the number of bins is determined automatically.
                - If a dictionary, expected format is {"bin_width": float}, where 'bin_width' is the width of each bin.

        Returns:
            tuple: A tuple containing the histogram values and bin edges.

        Note:
            This function concatenates the height data from multiple dimensions and plots the histogram.
            The number of bins can be set automatically or specified using a dictionary with the bin width.
        """
        ondDimData = np.concatenate(self.data)
        if isinstance(bins, dict):
            w  = bins["bin_width"]
            bins = np.arange(min(ondDimData), max(ondDimData) + w, w)
        hist = plt.hist(ondDimData, bins = bins)
        return hist

    def show_regions(self, 
                     figsize =(10,10), 
                     alpha = 0.65,
                     face_color = 'red',
                     rim_color = 'black', 
                    ):
        """
        Shows the regions as rectangular faces, wich were cut by cut_image_regions(window=int) so only feasebl for window cuts not vor voronoi!
        Parameters:
            figsize: tuple
                figure site in inc e.g. 10x10 inc
            alpha :float
                between 1 and 0 alpha of the reagions rectngulr
            face_color : string
                colors of rectangular (see matplolib colors). Default 'red'
            rim_color :
                colors of the rim(edge) of the faces, Default 'black'
        """
        plt.figure(figsize = figsize)
        fig = plt.axes()
        fig.imshow(self.data)
        for i in self.coor_regieons:
            #plt.scatter(i[0][:,0],i[0][:,1], color = 'red', alpha=0.5)
            x_min , x_max, y_min, y_max =min(i[0][:,0]), max(i[0][:,0]), min(i[0][:,1]), max(i[0][:,1])
            rectangle = plt.Rectangle((x_min,y_min), x_max - x_min, y_max - y_min, fc=face_color,ec=rim_color, alpha = alpha)
            fig.add_patch(rectangle)
            # break
        
        
    def plot_clusters_heights_distribution(self,
                                              bins = 10, 
                        bandwidth = 0.3E-9, 
                        figsize = (8,8), 
                        axfontsize = 10,
                        subtitle = False, 
                        sub_fontsize = 12,
                                 return_bining = False):
        """
        Plot the distribution of corrected heights for different cluster averaging methods.

        Args:
            bins (int): Number of bins for the histogram (default 10).
            bandwidth (float): Bandwidth for kernel density estimation (default 0.3E-9).
            figsize (tuple): Size of the figure (default (8, 8)).
            axfontsize (int): Font size for axis labels (default 10).
            subtitle (bool): Flag to include subtitles (default False).
            sub_fontsize (int): Font size for subtitles (default 12).
            return_bining (bool): Flag to return histogram binning data (default False).

        Returns:
            None or tuple: If return_bining is True, returns a tuple containing histogram binning data.

        Note:
            This function generates a 2x2 subplot with histograms and kernel density plots for the corrected heights
            obtained using different cluster averaging methods: average, closest step, and highest step.
        """
        fig = make_subplots(rows=2, cols=2,
                            horizontal_spacing = 0.15,
                            vertical_spacing = 0.15,
                            subplot_titles = (f"Heights with bins #={bins}",
                                             "",
                                            
                                              f"Kernel Density\n with bandwidth {bandwidth}",
                                             ""
                                            ))

        fig.update_layout(
        title_text= f'<b>{self.name}</b><br><b>{len(self.heights)}:</b> total # of clusters',
        autosize=False,
        width=figsize[0]*100,
        height=figsize[1]*100,
        # margin=dict(l=80, r=80, t=100, b=80)
        )

        
#         binning = plt.hist(self.heights['corrected_Z_averaged'], bins=bins,
#                         density=False)
        
#         binning2 = plt.hist(self.heights['corrected_Z_closest_step'], bins=bins,
#                         density=False)
        
#         binning3 = plt.hist(self.heights['corrected_Z_highest_step'], bins=bins,
#                         density=False)
        binning = np.histogram(self.heights['corrected_Z_averaged'], bins=bins,
                        density=False)
        
        binning2 = np.histogram(self.heights['corrected_Z_closest_step'], bins=bins,
                        density=False)
        
        binning3 = np.histogram(self.heights['corrected_Z_highest_step'], bins=bins,
                        density=False)
        color_average, color_closest_step, color_highest_step = ['black', 'red', 'green']
        
        bar_data = [
            go.Bar(y = binning[0],x=binning[1],
                   name ='Distribution average',
                   marker_color= color_average,
                   offsetgroup=1),
            
            go.Bar(y = binning2[0],
                   x=binning2[1], 
                   name ='Distribution closest step',
                   marker_color= color_closest_step,
                   offsetgroup=2,
                   opacity = 0.8
                  ),
            
            go.Bar(y = binning3[0],
                   x=binning3[1], 
                   name ='Distribution highest step',
                   marker_color= color_highest_step,
                   offsetgroup=2,
                   opacity = 0.8
                  )
        ]
        fig.add_traces(bar_data, rows=1, cols=1)
        fig.update_layout(barmode='overlay',
                          bargap=0.0,
                          # bargroupgap=0.0
                         )
        KernalPlot = self.heights['corrected_Z_averaged'].to_numpy()
        X_plot = KernalPlot[:, np.newaxis]
        kde = KernelDensity(#kernel="epanechnikov",
                            kernel="gaussian",
                             bandwidth=bandwidth ).fit(X_plot)
        log_dens = kde.score_samples(X_plot)
        
        KernalPlot2 = self.heights['corrected_Z_closest_step'].to_numpy()
        X_plot2= KernalPlot2[:, np.newaxis]
        kde2 = KernelDensity(#kernel="epanechnikov",
                            kernel="gaussian",
                             bandwidth=bandwidth ).fit(X_plot2)
        log_dens2 = kde.score_samples(X_plot2)
        
        KernalPlot3 = self.heights['corrected_Z_highest_step'].to_numpy()
        X_plot3= KernalPlot3[:, np.newaxis]
        kde3 = KernelDensity(#kernel="epanechnikov",
                            kernel="gaussian",
                             bandwidth=bandwidth ).fit(X_plot3)
        log_dens3 = kde.score_samples(X_plot3)
        
        deviation = self.heights['initial_Z'] - self.heights['corrected_Z_averaged']
        deviation2 = self.heights['initial_Z'] - self.heights['corrected_Z_closest_step']
        deviation3 = self.heights['initial_Z'] - self.heights['corrected_Z_highest_step']
        
        fig.add_trace(go.Scattergl(x=X_plot[:, 0],
                                   y=np.exp(log_dens),
                                   mode='markers',
                                   marker_color= color_average,
                                   name ='Kernel Density averaged'),
                      row=2, col=1)
        fig.add_trace(go.Scattergl(x=X_plot2[:, 0],
                                   y=np.exp(log_dens2),
                                   mode='markers',
                                   marker= dict(color = color_closest_step,
                                               opacity = 0.8),
                                   name ='Kernel Density closest step'),
                      row=2, col=1)
        fig.add_trace(go.Scattergl(x=X_plot3[:, 0],
                                   y=np.exp(log_dens3),
                                   mode='markers',
                                   marker= dict(color = color_highest_step,
                                               opacity = 0.8),
                                   name ='Kernel Density highest step'),
                      row=2, col=1)

        text = [f'Nr: {string1}<br>z_average: {Decimal(string2):.3E}' for string1, string2 in zip(self.heights.index.values, self.heights['corrected_Z_averaged'])]

        fig.add_trace(go.Scattergl( x = self.heights['initial_Z'], 
                                   y = deviation, 
                                   # text = self.heights.index.values,
                                   text = text,
                                   hoverinfo = 'text', 
                                   name = 'average', 
                                   mode = 'markers',
                                  marker=dict(color = color_average,
                                              symbol = 'cross')
                                  ),
                      row=1, col=2)
        text2 = [f'Nr: {string1}<br>z_closest_step: {Decimal(string2):.3E}' for string1, string2 in zip(self.heights.index.values, self.heights['corrected_Z_closest_step'])]
        
        fig.add_trace(go.Scattergl( x = self.heights['initial_Z'], 
                                   y= deviation2, 
                                   # text = self.heights.index.values,
                                   text = text2,
                                   hoverinfo = 'text', 
                                   name = 'closest step', 
                                   mode = 'markers',
                                   marker = dict(color = color_closest_step,
                                                 symbol = 'arrow-up',
                                                 opacity = 1)
                                  )
                      ,
                      row=1, col=2)
        text4 = [f'Nr: {string1}<br>z_heighest_step: {Decimal(string2):.3E}' for string1, string2 in zip(self.heights.index.values, self.heights['corrected_Z_highest_step'])]
        
        fig.add_trace(go.Scattergl( x = self.heights['initial_Z'], 
                                   y= deviation3, 
                                   # text = self.heights.index.values,
                                   text = text4,
                                   hoverinfo = 'text', 
                                   name = 'highest step', 
                                   mode = 'markers',
                                   marker = dict(color = color_highest_step,
                                                 symbol = 'arrow-down',
                                                 opacity = 1)
                                  )
                      ,
                      row=1, col=2)
        
        text3 = [f'Nr: {string1}<br>z_averaged: {Decimal(string2):.3E}<br>z_closest_step: {Decimal(string3):.3E}' for string1, string2, string3 in zip(self.heights.index.values,self.heights['corrected_Z_averaged'], self.heights['corrected_Z_closest_step'])]
        
        fig.add_trace(go.Scattergl(x = self.heights['initial_Z'], 
                                   y = self.heights['corrected_Z_averaged'] - self.heights['corrected_Z_closest_step'], 
                                    # text = self.heights.index.values,
                                    text = text3,
                                   hoverinfo = 'text', 
                                   name = 'average - closest step', 
                                   mode = 'markers',
                                   marker = dict(color = 'darkorange',
                                                 symbol = 'star',
                                                 )
                                  ),
                      row=2, col=2)
        
        
        
        text5 = [f'Nr: {string1}<br>z_averaged: {Decimal(string2):.3E}<br>z_highest_step: {Decimal(string3):.3E}' for string1, string2, string3 in zip(self.heights.index.values,self.heights['corrected_Z_averaged'], self.heights['corrected_Z_highest_step'])]
        
        fig.add_trace(go.Scattergl(x = self.heights['initial_Z'], 
                                   y = self.heights['corrected_Z_averaged'] - self.heights['corrected_Z_highest_step'], 
                                    # text = self.heights.index.values,
                                    text = text3,
                                   hoverinfo = 'text', 
                                   name = 'average - highest step', 
                                   mode = 'markers',
                                   marker = dict(color = 'black',
                                                 symbol = 'star',
                                                 )
                                  ),
                      row=2, col=2)
        
        fig['layout']['xaxis']['title']='Binned heights'
        fig['layout']['yaxis']['title']='Cluster per bin'
        fig['layout']['xaxis']['tickformat']= 'E'

        fig['layout']['xaxis2']['title']=f'Initial Z [{self.si_unit_z}]'
        fig['layout']['yaxis2']['title']='Initial_Z - Corrected_Z '
        fig['layout']['yaxis2']['tickformat']= 'E'
        fig['layout']['xaxis2']['tickformat']= 'E'
        #fig['layout']['yaxis2']['rangemode']= 'tozero'
        
        fig['layout']['yaxis3']['title']='Density distribution'
        fig['layout']['xaxis3']['title']=f'Z [{self.si_unit_z}]'
        fig['layout']['xaxis3']['tickformat']= 'E'
        
        fig['layout']['xaxis4']['title']=f'Initial Z [{self.si_unit_z}]'
        fig['layout']['yaxis4']['title']='Z_average - Z_step'
        fig['layout']['yaxis4']['tickformat']= 'E'
        fig['layout']['xaxis4']['tickformat']= 'e'
    
        fig.show()
        
    def calc_nearest_neighbor_distribution(self):
        """
        Calculate the distribution of nearest neighbor distances for each cluster.

        This method uses a KDTree to efficiently find the nearest neighbors for each point in the cluster.
        The nearest neighbor distances and corresponding points are stored in the class attribute
        'nearest_neighbors_distribution'.

        Returns:
        None
        """
        points = self.clusters_coord*np.array([self.xreal/self.xres,self.yreal/self.yres,1])
        # Create a KDTree
        tree = cKDTree(points)

        # List to store nearest neighbors for each point
        nearest_neighbors = []
        nearest_neighbors_distance = []
        # Loop through each point and find the nearest neighbor
        for query_point in points:
            distance, index = tree.query(query_point,k=2)
            distance, index = distance[1], index[1]
            nearest_neighbor = points[index]
            nearest_neighbors.append(nearest_neighbor)
            nearest_neighbors_distance.append(distance)
        self.nearest_neighbors_ditribution =  np.array(nearest_neighbors_distance)

def load_from_gwyddion(path : str) -> clusterpic: 
    """
    Creats list of objects from Gwyddion file by appling of clusterpic() for every pciture in gwyddion file 
    (so for UP, DOWN, FORWORD, BACKWARD)
    
    Returns:
        list of clusterpic objects
    """
    obj = gwyload(path)
    channels = get_datafields(obj)
    try:
        meta_data_dic = {v: obj[f'/{k}/meta'] for k, v in find_datafields(obj)} ## find all meta data 
    except Exception as e:
        meta_data_dic = {}
        warnings.warn('No meta data found.')
    objreturn ={}
    for i in channels.keys():
        try:
            MetaData =  pd.DataFrame.from_dict(meta_data_dic[i], orient= 'index')
        except Exception as e:
            MetaData = None
            warnings.warn('No meta data found.')
        
        objreturn[i] =  clusterpic(
                    path = path,
                    name = f'{channels[i]["xres"]}x{channels[i]["yres"]} pix {channels[i]["xreal"]:.2e}x{channels[i]["yreal"]:.2e} m',
                    data = channels[i].data,
                    xres = channels[i]['xres'],
                    yres = channels[i]['yres'],
                    xreal = channels[i]['xreal'],
                    yreal = channels[i]['yreal'],
                    si_unit_xy = channels[i]['si_unit_xy'],
                    si_unit_z = channels[i]['si_unit_z'],
                    metaData = MetaData
                )
        try:
            objreturn[i].currentSetPoint = float(objreturn[i].metaData.loc['EEPA:Regulator.Setpoint_1 [Ampere]'][0])
            objreturn[i].gapVoltage = float(objreturn[i].metaData.loc['EEPA:GapVoltageControl.Voltage [Volt]'][0])
        except Exception as e:
            warnings.warn(f'No gapVoltage or currentSetPoint data found. Check metaData: {e}')
    return objreturn

def load_from_pickle(path : str) -> clusterpic:
    """
    Loads saved clusterpic object frome pickle file in path
    """
    with open(path, "rb") as input_file:
            clusterpic_obj = pickle.load(input_file)
    input_file.close()
    return clusterpic_obj

def del_edge_clusters_by_pix(toProzess: clusterpic, deltPix: int = 0 ):
    """
    Delets clusters on the edge of the picture/dataset. Because it is probabely not the real highest spot
    
    Parameters:
        toProzess (clusterpic): STM Image clusterpic object where the edge cluster have to be removed
        deltPix: how many pixes from the edge are considerd vor deletation. E.g. deltPix = 3: 3Pisel strips on the edge of the Picter were every cluster within this stripe will be deletes
        
    """
    toProzess_t = toProzess.heights.T
    def delete_multiple_element(list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)
    indexes_to_delet = []            
    for index, row in toProzess.heights.iterrows():
        if (row['x'] >= toProzess.xres-deltPix) or  (row['y'] >= toProzess.yres-deltPix): 
            #print(index,row['x'], row['y'])
            indexes_to_delet.append(index)
            toProzess_t.pop(index)

        if (row['x'] <= 0 + deltPix) or  (row['y'] <= 0 + deltPix): 
            #print(index,row['x'], row['y'])
            toProzess_t.pop(index)
            indexes_to_delet.append(index)

    toProzess.heights = toProzess_t.T
    delete_multiple_element(toProzess.regions, indexes_to_delet) ## delet the regions with clusters on edge
    toProzess.clusters_coord = np.delete(toProzess.clusters_coord,indexes_to_delet, axis=0) ## delete edge cluster from coordinate list
    
def combine_heights(name:str = '', input_path: str = '', input_list: list = None, serach_patter: str = '') -> clusterpic:
    if input_list is None:
        input_list = listdir(input_path)
    result_dfs = []
    result_area = 0.0
    for i in input_list:
        if serach_patter == '':
            current_obj = load_from_pickle(input_path+i)
            result_dfs.append(current_obj.heights)
            result_area +=current_obj.area
        elif serach_patter in i: #clusterpic_obj
            current_obj = load_from_pickle(input_path+i)
            result_dfs.append(current_obj.heights)
            result_area += current_obj.area
            #print(result_area)
    result = pd.concat(result_dfs)
    blup = clusterpic(data = np.zeros((10,10)), name=name)
    blup.si_unit_xy = 'm'
    blup.si_unit_z = 'm'
    blup.heights = result
    blup.area = result_area
    return blup
