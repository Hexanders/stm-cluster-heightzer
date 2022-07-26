from  gwyfile import load as gwyload
from  gwyfile.util import get_datafields 
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import path ### just for "drawing" an polygon for later extraction of the values inside this polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from richdem import rdarray, TerrainAttribute

from io import StringIO 
import sys

import time

class Capturing(list):
    """
    Capture print() returns to the IO into a list
    
    Source: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    Usage:
        >>with Capturing() as output:
        >>    do_something(my_object)
        output: list with all prints in it
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
def load_from_gwyddion(path):
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
        objreturn.append(clusterpic(
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
        self.si_unit_xy = si_unit_xy
        self.si_unit_z = si_unit_z
        self.peak_XYdata = pd.DataFrame()
        self.clusters_coord = np.empty(0)
        self.ax = None
        self.pickable_artists = None
        self.event =None
        self.coor_regieons = []
        self.heights = pd.DataFrame()
        self.cluster_distribution = None
        
    def __repr__(self):
        return self.name
    
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
                dump_path = 's_cluster_coordinates.pkl' %(self.name)
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
        suspect = xyz_current_max
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
    def show_data(self):
        fig, ax = plt.subplots()
        ax.imshow(self.data, extent =[0, self.xreal, 0, self.yreal])
        
    def get_peakXYdata(self):
        """
        Calculates 2d numpy array of XY data of peaks estimatet by finde_peaks_in_rows().
            it is just the heigest points not the maxima of cluster, you need to run group_clusters() for this
        """
        return self.peak_XYdata
    
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
    
    def calc_cluster_distribution(self):
        """
        Calculates all distances c = sqrt((a1-a2)**2 + (b1-b2)**2) from calculated cluster heigts table. So calc_true_height_4_every_region() have to be run befor
        
        Returns:
        --------
            numpy.array:
                distace of every pare of all clusters
        """
        all_coord = np.array((self.heights['x[m]'],self.heights['y[m]'])).T # prepare 2d array vor surching distance
        distribution = []
        for i in range(0, len(all_coord)):
            for k in range(0, len(all_coord)):
                if k!=i:
                    distribution.append(np.sqrt( (all_coord[k][0] - all_coord[i][0])**2. 
                                            + ( all_coord[k][1]  - all_coord[i][1])**2. ))
        self.cluster_distribution = np.array(distribution)
        
    def cut_image_regions(self, window = None):
        """
        Cuts the image data in to regions around clusters determine by Voronoi algorithm if window id None. If window is given: cuts image in squers with the length of window
        Parameters:
            data: 2d numpy array, STM Image
            cluster_coordinates: Nx3 numpy array, ([x1,y2,z3], ...) the coordinates of clusters peeaks, z coordinate ist absolut
        Returns:
            list: [( Nx3 numpy.array, (x,y,z) ),...] list of all reagions with the regions data (Nx3 numpy.array) and coordinate of the custerpeak (x,y,z)
        """
        
        if window is not None:
            self.coor_regieons = []
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
                self.coor_regieons.append([np.array(slice_xyz),i]) 
        
        else:
            self.coor_regieons = []
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
            for i in regions: ##### Very Very slow need vectorization!!!!!!
                polygon_coord = path.Path(vertices[i]) ### extracting coordinates out of polygon
                a2d_coordinates_of_area = polygon_coord.contains_points(xy_data,radius=0.0)
                a3d_area = xyz_data[a2d_coordinates_of_area]
                dict_a3d_area = {}
                for i in a3d_area:
                     dict_a3d_area[(i[0],i[1])] = i[2]
              #  [(dict_a3d_area[(i[0],i[1])] = i[2]) for i in a3d_area]
                breakIt = False
                cluster_max_inthearea = None
                for i in self.clusters_coord:
                    if breakIt:
                        break
                    if (i[1],i[0]) in dict_a3d_area:
                        # cluster_max_inthearea =(i[0],i[1],dict_a3d_area[(i[0],i[1])])
                        cluster_max_inthearea =(i[0],i[1],i[2])
                        breakIt = True
                        break
                self.coor_regieons.append([a3d_area,np.array(cluster_max_inthearea)]) 
                #self.coor_regieons.update({idx:{'x_min_id_offset':x_range[0], 'y_min_id_offset':y_range[0],'slice':aslice,'xyz(max)':cluster_max_inthearea}})
                
    def find_groundlevel(self, 
                         region_data, 
                         slope_threshold_factor = 0.1, 
                         groundlevel_cutoff = 0.99, 
                         ):
        """
        Compute slope maps with richdem.TerrainAttribute and determine the ground level of the cluster for calculating of heights of the cluster due to this ground level  

        Parametes:
            region_data: list:
                list of [(numpy.array, (x,y,z))] region cuted by cut_image_regions() and the coordinates of maximum of cluster inside the region (x,y,z)
            slope_threshold_factor: float:
                e.g. 0.01 means 1% of max-min sloap value would be consiedert as posible ground level points
            groundlevel_cutoff: float:
                e.g. 0.3 means all values in ground level candidate points smaller than 30% of maximum of cluster would be consiedert as ground level
            seek_for_steps: string:
                if 'True' applay seek_steps_in_ground_level() and try to separate ground level higts
                if 'False' take average of ground level
                if 'both' calculate both and return tuple (average, seek_steps_in_ground_level())
       Returns:
            ground_level: list:
                list of x,y,z, all points in cuted area of cluster wich are belonging to ground level
        """
        x_min =  region_data[0][:,0].min() # region_data came from matplotlib.path and ist is not array, hier konvert to squer array with nan if no value. still empty array 
        y_min =  region_data[0][:,1].min()
        x_dim = np.arange(region_data[0][:,0].min(),region_data[0][:,0].max()+1)
        y_dim = np.arange(region_data[0][:,1].min(),region_data[0][:,1].max()+1)
        full_dim_array = np.empty((len(x_dim),len(y_dim)))
        full_dim_array[:] = np.NaN
        for i in range(0,full_dim_array.shape[0]): # fill empty array with data
            for j in range(0,full_dim_array.shape[1]):
                xxx = np.where((region_data[0][:,0] == i+x_min) & (region_data[0][:,1] == j+y_min) )
                if np.any(region_data[0][xxx[0]]):
                    full_dim_array[i][j] = region_data[0][xxx[0]][0][2]
        rda = rdarray(full_dim_array, no_data=-9999)  # calculate slope of the region
        with Capturing() as output:
            terr_data = TerrainAttribute(rda,
                                   attrib='slope_riserun'
                                  )
        threshhold = np.nanmin(terr_data) + (np.nanmax(terr_data)-np.nanmin(terr_data))*slope_threshold_factor### x% of difference from max to min
        flat_matrix = np.argwhere((terr_data<threshhold) & (terr_data.any()))

        correct_flat_matrix = np.vstack((flat_matrix[:,0]+x_min, flat_matrix[:,1]+y_min)).T ### correct for the rigth coordinates due to the comleet picture
        xyz_flat_matrix = []
        #start_time = time.time()
        for i in region_data[0]:
            for k in correct_flat_matrix:
                if (k[0] == i[0]) & (k[1] == i[1]):
                    xyz_flat_matrix.append([i[0],i[1],i[2]])
        xyz_flat_matrix = np.asarray(xyz_flat_matrix)

        y_max,x_max,z_max = region_data[1]
        ground_level = np.array([])
        while len(ground_level) <= 1:
            ground_level = xyz_flat_matrix[np.where(xyz_flat_matrix[:,2]<(z_max-(z_max-xyz_flat_matrix[:,2].min())*groundlevel_cutoff))]
            groundlevel_cutoff = groundlevel_cutoff - 0.01
            if groundlevel_cutoff <= 0.90:
                break
        return ground_level
    
    def true_hight(self, region_data, 
                         slope_threshold_factor = 0.01, 
                         groundlevel_cutoff = 0.30, 
                         method='complete',
                         metric='minkowski',
                         threshold = 'default', 
                         thold_default_factor = 1.1,
                         cutoff_points =  5,
                         seek_for_steps = 'False'):
        """
        Correct the heights in z for specific reagion_data (witch were cuted by cut_image_regions)
        
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
                default = 1.1
                
            cutoff_points: int 
                number of points in a ground_level clusterd points, witch would be considered as artifacte e.g. 'jumping' stm tip/nois
                default is 5. But you may set it iven to 0 (all points then includet)
                
            seek_for_steps: string 
                if 'False' correction of z value only by avareging all point in ground level
                if 'True' correction of z value only by avareging all point in nearst cluster of points in ground level
                if 'both' correct both
                defalut = 'False'
        Returns:
                list with 4 or 5 entries 
                if seek_for_steps = 'False' 
                    ['x','y','z', 'z-z_avarage(nearest_step)']
                if seek_for_steps = 'True' 
                    ['x','y','z', 'z-z_average(ground_level)']
                if seek_for_steps = 'both'
                    ['x','y','z', 'z-z_average(ground_level)', 'z-z_avarage(nearest_step)']   
        """
        ground_level = self.find_groundlevel(region_data, 
                         slope_threshold_factor = slope_threshold_factor, 
                         groundlevel_cutoff = groundlevel_cutoff)
        z_max = region_data[1][2]
        if seek_for_steps == 'True':
            true_hight = [z_max - self.seek_steps_in_ground_level(ground_level,
                                                                  region_data[1],
                                                                  method=method,
                                                                  metric=metric,
                                                                  threshold = threshold,
                                                                  thold_default_factor = thold_default_factor,
                                                                  cutoff_points = cutoff_points
                                                                 )]
        elif seek_for_steps == 'False':
            true_hight = [z_max - np.average(ground_level[:,2])]
        elif seek_for_steps == 'both':
            true_hight = [z_max - np.average(ground_level[:,2]),
                          z_max - self.seek_steps_in_ground_level(ground_level,
                                                                  region_data[1],
                                                                  method=method,
                                                                  metric=metric,
                                                                  threshold = threshold,
                                                                  thold_default_factor = thold_default_factor,
                                                                  cutoff_points = cutoff_points
                                                                 )]
            
        return true_hight
    
    def seek_steps_in_ground_level(self, ground_level,
                                   peak_coordinats,
                                   method='complete',
                                   metric='minkowski',
                                  threshold = 'default',
                                  thold_default_factor = 1.1,
                                  cutoff_points = 5):
        """
        Accumulate points of the ground level points (points without steep slope) 
        in to clusters (see scipy.cluster.hierarchy)
        
        Parameters:
            ground_level: numpy array: 
                XYZ data . E.g ground_level[:,0] := 'X column'
            method: str: 
                method used by scipy.cluster.hierarchy.linkage
                default is 'complete'
            metric: str:
                metric used by scipy.cluster.hierarchy.linkage
                default is 'minkowski'
            threshold: str or 
                metric used by scipy.cluster.hierarchy.linkage
                if default a squear root of (x_min - x_max)**2 + (y_min-y_max) times thold_default_factor
                where x_ , y_ are coordinates 
            thold_default_factor: float:
                see thershold
                default 1.1 so 110%
            cutoff_points: int 
                number of points in a ground_level clusterd points, witch would be considered as artifacte e.g. 'jumping' stm tip/nois
                default is 5. But you may set it iven to 0 (all points then includet)
                
        """
        factor = max(ground_level[:,0])/max(ground_level[:,2])  # normalization factor for Z for scipy.cluster.hierarchy.fcluster otherwise clustering of groundlevel points is not working in Z direction
        ground_level[:,2] =ground_level[:,2]*factor  #normalize Z

        Z = linkage(ground_level,
                method=method,  # dissimilarity metric: max distance across all pairs of 
                                        # records between two clusters
                        metric=metric
                )                           
        if threshold != 'default':
            t = threshold
        else:
            t = np.sqrt(abs(max(ground_level[:,0])-min(ground_level[:,0]))**2.+ 
                    abs(max(ground_level[:,1])-min(ground_level[:,1]))**2.)*thold_default_factor  # default threshold   
        clusters = fcluster(Z, t, criterion='distance',depth = 5) 
        all_clusters_list = [ground_level[np.where(clusters==k)[0].tolist()] for k in np.unique(clusters)]

        ground_level[:,2] =ground_level[:,2]/factor
        for j in all_clusters_list:
            j[:,2] = j[:,2]/factor
        min_distance = None
        n = cutoff_points # cut off criterium for quantity of points inside an clustered ground_level, e.g. artifactes of 'jumping' stm tip
        counter = 0
        
        subblot = plt.subplots(2, 2)
        ((ax1, ax2), (ax3, ax4)) = subblot[1]
        fig = subblot[0]
        ax4.scatter(ground_level[:,0],ground_level[:,1])
        for i in all_clusters_list:
            if len(i[:,0]) <=n: # Eliminate some artifacts, wenn the clustered ground_level has les then n points
                continue
            mean_x, mean_y = ((max(i[:,0])-min(i[:,0]))/2)+min(i[:,0]),((max(i[:,1])-min(i[:,1]))/2)+min(i[:,1])
            distance = np.sqrt(abs(mean_x - peak_coordinats[0])**2. + abs(mean_y-peak_coordinats[1])**2.)
            if min_distance is None:  # finde smallest distance in xy to the cluster center
                min_distance = (distance, counter)
            if min_distance[0] > distance:
                min_distance = (distance, counter)
            counter +=1
            
            ax4.scatter(mean_x, mean_y, s=80, label = 'd:%s'%round(distance,2))
            ax4.scatter(peak_coordinats[0],peak_coordinats[1],  marker = 'x', label  ='center' )
            ax3.scatter(i[:,0],i[:,2])
            ax2.scatter(i[:,1],i[:,2],alpha=1)
            
        try:
            avaraged_heigt_of_closest_groundlevel = np.average(
                all_clusters_list[min_distance[1]][:,2])
        except TypeError as err:
            print ("TypeError: {0} \nProbably the value for cutoff_points is set too high. Try smaller one ore even 0".format(err))
            raise
        
        
        
        return avaraged_heigt_of_closest_groundlevel
    
    def calc_true_height_4_every_region(self, 
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
        hights = []
        for i in self.coor_regieons:
            result = self.true_hight(i, 
                         slope_threshold_factor = slope_threshold_factor , 
                         groundlevel_cutoff = groundlevel_cutoff, 
                         method=method,
                         metric=metric,
                         threshold = threshold, 
                         thold_default_factor = thold_default_factor,
                         cutoff_points = cutoff_points,
                         seek_for_steps = seek_for_steps)
            result2 = [i[1][0], i[1][1], (i[1][0]/self.xres)*self.xreal, (i[1][1]/self.yres)*self.yreal, i[1][2]]
            result2.extend(result)
            hights.append(result2)
            
            # print(hights)
            break
        if seek_for_steps == 'True': 
            columns = ['x[pix]','y[pix]','x[m]','y[m]','z', 'z-z_avarage(nearest_step)']
        elif seek_for_steps == 'False':
            columns = ['x[pix]','y[pix]','x[m]','y[m]','z', 'z-z_average(ground_level)']  
        elif seek_for_steps == 'both':               
            columns = ['x[pix]','y[pix]','x[m]','y[m]','z', 'z-z_average(ground_level)', 'z-z_avarage(nearest_step)']
        print(hights)
        self.heights = pd.DataFrame(hights, columns = columns)
        # self.heights = np.array(hights)
        # self.heights = [np.array(hights), columns]
        
    def cluster_peaker(self,
                       voronoi_pattern = True, 
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
        vor = Voronoi( np.vstack((self.clusters_coord[:,0], self.clusters_coord[:,1])).T, qhull_options='Qbb Qc Qx')
        if voronoi_pattern:
            voronoi_plot_2d(vor, ax = ax, show_points = False)
        ax.imshow(self.data)
        if show_regions:
            for i in self.coor_regieons:
                x_min , x_max, y_min, y_max =min(i[0][:,0]), max(i[0][:,0]), min(i[0][:,1]), max(i[0][:,1])
                rectangle = plt.Rectangle((x_min,y_min), x_max - x_min, y_max - y_min, fc=face_color,ec=rim_color, alpha = alpha)
                ax.add_patch(rectangle)
        #global pickable_artists
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
        return ax, pickable_artists
    
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
        
     
