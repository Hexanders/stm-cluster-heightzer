from .Regions import *
from  gwyfile import load as gwyload
from  gwyfile.util import get_datafields 
import pickle
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from matplotlib import path ### just for "drawing" an polygon for later extraction of the values inside this polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing 
from sklearn.neighbors import KernelDensity
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
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
                 xreal = None, yreal= None, si_unit_xy= None, si_unit_z= None):
        self.path = path,
        self.name = name
        self.data = data 
        self.xres = xres 
        self.yres = yres 
        self.xreal = xreal 
        self.yreal = yreal 
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
        self.heights = pd.DataFrame(columns = ['x',
                                               'y',
                                               f'x_{self.si_unit_xy}',
                                               f'y_{self.si_unit_xy}',
                                               'initial_Z', 
                                               'corrected_Z_averaged', 
                                               'corrected_Z_closest_step',
                                              'corrected_Z_highest_step'])
        self.cluster_distribution = None
        
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

    
    def walk_to_the_max(self,test_data, xyz_current_max, pixelRange = 2):
        """
        Finds locle maxima by slicing test_data (NXN array) here STM image data with a window of +- PixelRange in first and second dimension. 
        It searches for local maxima in the slices and if there is no other maxima the search is aborted.

        Parameters:

        test_data: NxN numpy array, STM image data
        xyz_current_max: list of [x,y,z] coordinates of maximum/point from witch start searching
        pixelRange: integer, how big is the wind in wich to search. E.g pixelRange=4 produces 8X8 window (array with the shape = (8,8))
        """
        no_new_max_found = True
        suspect = xyz_current_max#[1],xyz_current_max[0], xyz_current_max[2]
        step_counter = 0
        while no_new_max_found:
            step_counter +=1
            y_range = [suspect[0]-pixelRange, suspect[0]+pixelRange]
            if y_range[0] < 0:y_range[0] = 0 # if you hit the boundaries important for correction later see maxXX
            if y_range[1] > test_data.shape[0] : y_range[1] = test_data.shape[0] # if you hit the boundaries 
            x_range = [suspect[1]-pixelRange, suspect[1]+pixelRange] # if you hit the boundaries
            if x_range[0] < 0 : x_range[0] = 0  # if you hit the boundaries
            if x_range[1] > test_data.shape[1] : x_range[1] = test_data.shape[1] # if you hit the boundaries
            aslice = test_data[x_range[0]:x_range[1],y_range[0]:y_range[1]]
            maxX, maxY = np.unravel_index(aslice.argmax(), aslice.shape) ## finde maximum ids in 2d array slice
            maxXX, maxYY = maxX+x_range[0], maxY+y_range[0] ## correct for the actual array, so not the slice
            #self.tmp.append([aslice.max() ,suspect])
            if np.isnan(suspect[2]): ### some times it is just nan, no idea why. This is not very good fix but it works
                suspect[2] = 0.0
            if aslice.max() > suspect[2]:
                suspect = [maxYY, maxXX, aslice.max() ]
                
                #no_new_max_found = False
            else:
                no_new_max_found = False

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
                  bar = True,
                  bar_space_left = 0.05, # space in % from hole range
                  bar_space_bottom = 0.95,# space in % from hole range
                  bar_length = None, # if non will be calculatet automaticaly 10% of picture width
                  bar_color = 'white',
                  bar_size = 10,
                  bar_label_xshift = 1.5,   # in percent from origin
                  bar_label_yshift = 0.99,  # in percent from origin
                  bar_ticks = False,
                  unit = 'nm',
                  no_ticks =False,
                  show_clusters = False,
                  clusters_markersize = 3,
                  font_size= 10,
                  ):
        """
    Plots the scanning tunneling microscope (STM) data.

    Parameters:
    -----------
    cmap: str
        The colormap used for the plot. Default is 'gray'.
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
        im = ax.imshow(self.data,
                   cmap = cmap,
                   #origin = 'lower',
                   extent =[0, self.xreal, self.yreal, 0]
                   )
        if show_clusters:
            for i in range(0,len(self.clusters_coord)):
                ax.plot(self.clusters_coord[:,0][i]*(self.xreal/self.xres),
                    self.clusters_coord[:,1][i]*(self.yreal/self.yres),
                        'o', c = 'r', ms = clusters_markersize)
        if bar:
            ax.hlines(self.yreal*bar_space_bottom,
                       #1E-8,
                       xmin= self.xreal*bar_space_left,
                       xmax = self.xreal*bar_space_left + bar_length*1e-9,
                       colors = bar_color,
                       linewidth = bar_size)
    
            ax.annotate(str(bar_length)+' '+unit,
                    (self.xreal*bar_space_left*bar_label_xshift,self.yreal*bar_space_bottom*bar_label_yshift),
                             
                             color = bar_color)
        if unit == 'nm':
            func = lambda x,pos: "{:g}".format(x*1e9)
        if unit == '$\mu$m':
            func = lambda x,pos: "{:g}".format(x*1e6)
        if unit == '$\AA$':
            func = lambda x,pos: "{:g}".format(x*1e10)
        
        
        fmt = mpl_ticker.FuncFormatter(func)

        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # axins = inset_axes(ax,
        #             width="1%",  
        #             height="100%",
        #             bbox_to_anchor=(1,0.5),
        #             # loc='lower center',
        #             #borderpad=-5
        #            )
        cbar = plt.colorbar(im,
                            #cax=axins,
                            format = fmt,
                            pad = 0.01)
        
        cbar.ax.set_title(unit,
                          loc= 'right',
                          pad = 0)
        #cbar.ax.set_ylabel(unit)
        if bar_ticks == False: cbar.ax.tick_params(size = 0, pad =0.3)
        if no_ticks:
            # Hide X and Y axes label marks
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            ax.set_xticks([])
            ax.set_yticks([])
        #plt.tight_layout()
        #plt.subplots_adjust(wspace=0.01)
        
            
        return ax
    
    def get_peakXYdata(self):
        """
        Calculates 2d numpy array of XY data of peaks estimatet by finde_peaks_in_rows().
            it is just the heigest points not the maxima of cluster, you need to run group_clusters() for this
        """
        return self.peak_XYdata
    
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
        
    def update_peaked_clusters(self, pickable_artists, xyz =None, max_crawler = False):
        """
        Updates the list of clusters withch was peakt by cluster_peaker() or by given list xyz
        Parameters:
            data: 2d numpy array, STM Image
            pickable_artists: matplotlib.pickable_artists
            xyz: [x,y,z] or list of [[x1,y1,z1], [x2,y2,z2], ...] coordinates if something gets wrong with the UI peaking cluster
            max_crawler: If True an xy is given, uses walk_to_the_max() for finding maximum
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
                        new_xyz.append(np.array(self.walk_to_the_max(self.data,i)[0]))
                    else:
                        new_xyz.append(i)
                clusters_coord = np.vstack((clusters_coord,np.array(new_xyz)))
            else:
                if max_crawler:
                    new_xyz = self.walk_to_the_max(self.data,xyz)[0]
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
        
        Returns:
        --------
            numpy.array:
                distace of every pare of all clusters
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
        
        Returns:
        --------
            numpy.array:
                distace of every pare of all clusters
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
        
    def cut_image_regions(self, window = None):
        """
        Cuts the image data in to regions around clusters determine by Voronoi algorithm if window id None. If window is given: cuts image in squers with the length of window
        
        Args:
            window (int): length of a side of a rectangular for the window bilding arraound a cluster 
        """
        
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
                a_region.cluster_peak_coordinates = np.array(i)
                a_region.region_type = region_type
                self.regions.append(a_region)
        else:
            self.regions = []
            region_type = 'voronoi'
            self.heights.index.name = region_type
            vor = Voronoi( np.vstack((self.clusters_coord[:,0], self.clusters_coord[:,1])).T, qhull_options='Qbb Qc Qx')
            regions, vertices = self.voronoi_finite_polygons_2d(vor)### conver regions in to finit regeons
            coor_regieons = []
            xyz_data = []
            for iy, ix in np.ndindex(self.data.shape): ###Reshaping NxN array int too 3XN .shoulb be vectorized!!!! 
                xyz_data.append([ix,iy,self.data[ix,iy]])
            #[xyz_data.append([ix,iy,data[ix,iy]]) for iy, ix in np.ndindex(data.shape)] ### slower!!
            xyz_data = np.array(xyz_data)
            #start_time = time.time()
            xy_data = np.array([xyz_data[:,1],xyz_data[:,0]]).T ### reduce from 3XN to 2XN needed for matplotlib.path.path.contains_points
            for idx,i in enumerate(regions): ##### Very Very slow need vectorization!!!!!!
                polygon_coord = path.Path(vertices[i]) ### extracting coordinates out of polygon
                a2d_coordinates_of_area = polygon_coord.contains_points(xy_data,radius=0.0)
                a3d_area = xyz_data[a2d_coordinates_of_area]
                dict_a3d_area = {}
                for i in a3d_area:
                     dict_a3d_area[(i[0],i[1])] = i[2]
              #  [(dict_a3d_area[(i[0],i[1])] = i[2]) for i in a3d_area]
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
                a_region.cluster_peak_coordinates = np.array(cluster_max_inthearea)
                a_region.region_type = region_type
                self.regions.append(a_region) 
                #self.coor_regieons.update({idx:{'x_min_id_offset':x_range[0], 'y_min_id_offset':y_range[0],'slice':aslice,'xyz(max)':cluster_max_inthearea}})
                
   
    
    def parralel_correct_height(self, 
                        slope_threshold_factor = 0.1, 
                         groundlevel_cutoff = 0.30, 
                         method='complete',
                         metric='minkowski',
                         threshold = 'default', 
                         thold_default_factor = 1.1,
                         cutoff_points = 5,
                         seek_for_steps = 'False'):
        """
        Calculates the hights in parralel
        """
        for region in self.regions: #set some variales
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
        self.heights = self.heights[0:0] #empty heigths table
        for idx,i in enumerate(self.regions):# collecting corrected hights
            self.update_height(i, idx)
            
    def correct_heights_4_regions(self, i):
        """
        just needed fo parralel_correct_height()
        """
        i.find_groundlevel()
        i.calc_true_hight()
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
                       show_regions = False,
                       face_color = 'red',
                       rim_color = 'black', 
                       alpha = 0.8):
        """
        Plots Cluster coordinates over data and let you delet and add cluster position by mouse click

        Parameter:
            data: 2d numpy array, STM Image
            cluster_coordinates: nX3 numpy arra line ([ x1,y1,z1],[x2,y2,z2], ...), xyz coordinates of prefound clusters by finde_peaks_in_rows()
            !! Atention!! This function defindes global peakable_artist variable, witsch you can am must use outside this function. 
                Otherwisse the hand peaking of clusters wuld not work
            voronoi_pattern: boolean, if True Voronoi pattern are showen in picture
        Returns: 
            matplotlib.axes
            pickable_artists: list of x,y,z cluster coordinates, if some werde deleted or added differs from cluster_coordinates

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
        ax.imshow(self.data)
        
        pickable_artists = []
        for i in range(0,len(self.clusters_coord)):
            pt, = ax.plot(self.clusters_coord[:,0][i],self.clusters_coord[:,1][i], 'o', c = 'r', ms = markersize)  
            pickable_artists.append(pt)

        coords = []
        new_maxes =[]

        removable = []
    
        def onclick(event,pickable_artists,gwy_data):
            if event.inaxes is not None and not hasattr(event, 'already_picked'):
                ax = event.inaxes

                remove = [artist for artist in pickable_artists if artist.contains(event)[0]]

                if not remove:
                    # add a pt        
                    x, y = ax.transData.inverted().transform_point([event.x, event.y])
                    new_max = self.walk_to_the_max(self.data,[int(x),int(y),self.data[int(x)][int(y)]],pixelRange=pixelRange)
                    pt, = ax.plot(new_max[0][0], new_max[0][1], 'o', picker=pikerRange, c='r')
                    pickable_artists.append(pt)
                    removable.append(new_max)

                else:
                    removable.append(remove)
                    pickable_artists.remove(remove[-1])
                    for artist in remove:
                        artist.remove()
                plt.draw()
        self.cid = fig.canvas.mpl_connect('button_press_event',lambda event: onclick(event,pickable_artists,self.data))

        # ax.set_xlim(vor.min_bound[0] - 10, vor.max_bound[0] + 10)
        # ax.set_ylim(vor.min_bound[1] - 10, vor.max_bound[1] + 10)

        ax.set_xlim(0, self.data.shape[1])
        ax.set_ylim(0, self.data.shape[0])

        if cluster_numbers:
            for i in range(0,len(self.clusters_coord)):
                ax.annotate(i, (self.clusters_coord[:,0][i], self.clusters_coord[:,1][i]), fontsize=8)

        ax.set_title('Total clusters: '+str(len(self.clusters_coord)))
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()                     # and move the X-Axis
        fig.set_facecolor('white')
        return ax, pickable_artists
    
    def update_height(self, region, index):
        y,x,z = region.cluster_peak_coordinates
        x_m = x*self.xreal/self.xres
        y_m = y*self.yreal/self.yres
        true_height = region.true_hight
        true_height_closest= region.true_hight_closest_ground_level
        true_height_heighest= region.true_hight_heighest_ground_level
        # self.heights([x,y,z, true_height, true_height_closest], columns = ['x','y','initial z', 'corrected Z averaged', 'corrected Z closest step'], index = index)
        self.heights.loc[index] = [x,y,x_m,y_m,z, true_height, true_height_closest,true_height_heighest]
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
        
        
    def plot_heights_distribution(self,
                                              bins = 10, 
                        bandwidth = 0.3E-9, 
                        figsize = (8,8), 
                        axfontsize = 10,
                        subtitle = False, 
                        sub_fontsize = 12,
                                 return_bining = False):
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
        
def load_from_gwyddion(path : str) -> clusterpic: 
    """
    Creats list of objects from Gwyddion file by appling of clusterpic() for every pciture in gwyddion file 
    (so for UP, DOWN, FORWORD, BACKWARD)
    
    Returns:
        list of clusterpic objects
    """
    obj = gwyload(path)
    channels = get_datafields(obj)
    objreturn =[]
    for i in channels.keys():
        #print(channels[i])
        objreturn.append(
            clusterpic(
                    path = path,
                    name = i,
                    data = channels[i].data,
                    xres = channels[i]['xres'],
                    yres = channels[i]['yres'],
                    xreal = channels[i]['xreal'],
                    yreal = channels[i]['yreal'],
                    si_unit_xy = channels[i]['si_unit_xy'],
                    si_unit_z = channels[i]['si_unit_z']
                ))
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
