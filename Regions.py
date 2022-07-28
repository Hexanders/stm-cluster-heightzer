from dataclasses import dataclass, field
import numpy as np 
from richdem import rdarray, TerrainAttribute
import sys
from io import StringIO 
import copy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from matplotlib import cm

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

@dataclass
class region():
    region_id: int = None
    region_type: str = ''
    coordinates : list[np.ndarray] = field(default_factory = list)
    cluster_peak_coordinates : np.ndarray =np.empty(0)
    ground_level : np.ndarray =np.empty(0)
    ground_level_regions : list[np.ndarray] = field(default_factory = list)
    true_hight : float = None
    true_hight_closest_ground_level : float = None
    slope_threshold_factor: float = 0.1 
    groundlevel_cutoff: float = 0.30 
    method: str ='complete'
    metric: str='minkowski'
    threshold: str = 'default'
    thold_default_factor: float = 1.1
    cutoff_points: int =  5
    seek_for_steps: bool = False

    def find_groundlevel(self):
        """
        Compute slope maps with richdem.TerrainAttribute and determine the ground level of the cluster for calculating of heights of the cluster due to this ground level  

       Returns:
            ground_level: list:
                list of x,y,z, all points in cuted area of cluster wich are belonging to ground level
        """
        x_min =  self.coordinates[:,0].min() # region_data came from matplotlib.path and ist is not array, hier konvert to squer array with nan if no value. still empty array 
        y_min =  self.coordinates[:,1].min()
        x_dim = np.arange(self.coordinates[:,0].min(),self.coordinates[:,0].max()+1)
        y_dim = np.arange(self.coordinates[:,1].min(),self.coordinates[:,1].max()+1)
        full_dim_array = np.empty((len(x_dim),len(y_dim)))
        full_dim_array[:] = np.NaN
        for i in range(0,full_dim_array.shape[0]): # fill empty array with data
            for j in range(0,full_dim_array.shape[1]):
                xxx = np.where((self.coordinates[:,0] == i+x_min) & (self.coordinates[:,1] == j+y_min) )
                if np.any(self.coordinates[xxx[0]]):
                    full_dim_array[i][j] = self.coordinates[xxx[0]][0][2]
        rda = rdarray(full_dim_array, no_data=-9999)  # calculate slope of the region
        with Capturing() as output:
            terr_data = TerrainAttribute(rda,
                                   attrib='slope_riserun'
                                  )
        threshhold = np.nanmin(terr_data) + (np.nanmax(terr_data)-np.nanmin(terr_data))*self.slope_threshold_factor### x% of difference from max to min
        flat_matrix = np.argwhere((terr_data<threshhold) & (terr_data.any()))

        correct_flat_matrix = np.vstack((flat_matrix[:,0]+x_min, flat_matrix[:,1]+y_min)).T ### correct for the rigth coordinates due to the comleet picture
        xyz_flat_matrix = []
        #start_time = time.time()
        for i in self.coordinates:
            for k in correct_flat_matrix:
                if (k[0] == i[0]) & (k[1] == i[1]):
                    xyz_flat_matrix.append([i[0],i[1],i[2]])
        xyz_flat_matrix = np.asarray(xyz_flat_matrix)

        y_max,x_max,z_max = self.cluster_peak_coordinates
        ground_level = np.array([])
        while len(ground_level) <= 1:
            ground_level = xyz_flat_matrix[np.where(xyz_flat_matrix[:,2]<(z_max-(z_max-xyz_flat_matrix[:,2].min())*self.groundlevel_cutoff))]
            groundlevel_cutoff = self.groundlevel_cutoff - 0.01
            if groundlevel_cutoff <= 0.90:
                break
        self.ground_level = ground_level
        return self
        
    def calc_true_hight(self):
        """
            Correct the heights in z for specific reagion_data (witch were cuted by cut_image_regions)

           
            Returns:
                    list with 4 or 5 entries 
                    if seek_for_steps = 'False' 
                        ['x','y','z', 'z-z_avarage(nearest_step)']
                    if seek_for_steps = 'True' 
                        ['x','y','z', 'z-z_average(ground_level)']
                    if seek_for_steps = 'both'
                        ['x','y','z', 'z-z_average(ground_level)', 'z-z_avarage(nearest_step)']   
            """
        z_max = self.cluster_peak_coordinates[2]
        if self.seek_for_steps:
            self.true_hight_closest_ground_level = self.seek_steps_in_ground_level()
            self.true_hight = z_max - np.average(self.ground_level[:,2])
                
        elif self.seek_for_steps == False:
            self.true_hight = z_max - np.average(self.ground_level[:,2])
        return self
            
    def seek_steps_in_ground_level(self):
        """
        Accumulate points of the ground level points (points without steep slope) 
        in to clusters (see scipy.cluster.hierarchy)
        """
        ground_level = copy.deepcopy(self.ground_level)
        factor = max(ground_level[:,0])/max(ground_level[:,2])  # normalization factor for Z for scipy.cluster.hierarchy.fcluster otherwise clustering of groundlevel points is not working in Z direction
        
        ground_level[:,2] =ground_level[:,2]*factor  #normalize Z

        Z = linkage(ground_level,
                method=self.method,  # dissimilarity metric: max distance across all pairs of 
                                        # records between two clusters
                        metric=self.metric
                )  
        if self.threshold != 'default':
            t = self.threshold
        else:
            t = np.sqrt(abs(max(ground_level[:,0])-min(ground_level[:,0]))**2.+ 
                    abs(max(ground_level[:,1])-min(ground_level[:,1]))**2.)*self.thold_default_factor  # default threshold   
        clusters = fcluster(Z, t, criterion='distance',depth = 5) 
        self.ground_level_regions = [ground_level[np.where(clusters==k)[0].tolist()] for k in np.unique(clusters)]

        for j in self.ground_level_regions:
            j[:,2] = j[:,2]/factor
        min_distance = None
        n = self.cutoff_points # cut off criterium for quantity of points inside an clustered ground_level, e.g. artifactes of 'jumping' stm tip
        counter = 0
        
        
        for i in self.ground_level_regions:
            if len(i[:,0]) <=n: # Eliminate some artifacts, wenn the clustered ground_level has les then n points
                continue
            mean_x, mean_y = ((max(i[:,0])-min(i[:,0]))/2)+min(i[:,0]),((max(i[:,1])-min(i[:,1]))/2)+min(i[:,1])
            distance = np.sqrt(abs(mean_x - self.cluster_peak_coordinates[0])**2. + abs(mean_y - self.cluster_peak_coordinates[1])**2.)
            if min_distance is None:  # finde smallest distance in xy to the cluster center
                min_distance = (distance, counter)
            if min_distance[0] > distance:
                min_distance = (distance, counter)
            counter +=1
            
            
        try:
            avaraged_heigt_of_closest_groundlevel = self.cluster_peak_coordinates[2] - np.average(
                self.ground_level_regions[min_distance[1]][:,2])
        except TypeError as err:
            avaraged_heigt_of_closest_groundlevel = None                
        
        
        return avaraged_heigt_of_closest_groundlevel
    
    def plot_ground_level(self, 
                          figsize = (10,8),
                         axis_view = None,
                         show_all_ground_level_points = True,
                         saveimage =None,
                         saveprefix ='',
                         dpi=100):
        """
        Plots level ground of a region and there regions witch could be separatet as different level
        """
        subblot = plt.subplots(2, 2)
        ((ax1, ax2), (ax3, ax4)) = subblot[1]
        fig = subblot[0]
        fig.set_size_inches(figsize)
        X,Y,Z = self.ground_level[:,0],self.ground_level[:,1],self.ground_level[:,2]
        
        ax3.scatter(X,Z)
        ax3.set_xlabel('x coordinate')
        ax3.set_ylabel('Z coordinate')

        ax2.scatter(Y,Z,
                    alpha = 1,
                    s=0.1)
        ax2.set_xlabel('y coordinate')
        ax2.set_ylabel('Z coordinate')
        if show_all_ground_level_points:
            ax4.scatter(X,Y)
        ax4.set_xlabel('x coordinate')
        ax4.set_ylabel('y coordinate')
        ax1.remove()
        ax1 = fig.add_subplot(221,projection='3d')
        if axis_view:
            ax1.view_init(axis_view[0],axis_view[1])
        X,Y,Z = self.coordinates[:,0],self.coordinates[:,1],self.coordinates[:,2]
        ax1.plot_trisurf(X, Y, Z,  linewidth=0.1, cmap=cm.jet, alpha = 0.8)
        #ax1.set_xticklabels('')
        #ax1.set_yticklabels('')
        ax1.set_zticklabels('')
        ax1.dist = 8
        counter = 0
        min_distance = None
        for Grcounter,i in enumerate(self.ground_level_regions):
            if len(i[:,0]) <=self.cutoff_points: # Eliminate some artifacts, wenn the clustered ground_level has les then n points
                continue
            sc = ax4.scatter(i[:,0],i[:,1],  s = 2 , label = 'Group:%s' %Grcounter)
            colors = sc.get_facecolor()
            mean_x, mean_y = ((max(i[:,0])-min(i[:,0]))/2)+min(i[:,0]),((max(i[:,1])-min(i[:,1]))/2)+min(i[:,1])
            distance = np.sqrt(abs(mean_x - self.cluster_peak_coordinates[0])**2. + abs(mean_y-self.cluster_peak_coordinates[1])**2.)
            if min_distance is None:  # finde smallest distance in xy to the cluster center
                min_distance = (distance, counter)
            if min_distance[0] > distance:
                min_distance = (distance, counter)
            ax4.scatter(mean_x, mean_y, 
                        s=80,
                        color = colors,
                        label = 'd:%s'%round(distance,2))
            ax3.scatter(i[:,0],i[:,2],
                        color = colors,
                       )
            ax2.scatter(i[:,1],i[:,2],
                        color = colors,
                        alpha=1)
            counter +=1
        ax4.scatter(self.cluster_peak_coordinates[0],
                        self.cluster_peak_coordinates[1],  
                        marker = 'x', 
                        label  ='center' )
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.suptitle(f'Region Nr: {self.region_id}')
        plt.tight_layout()
        if saveimage:
            fig.savefig('%s_region%s.%s' %(saveprefix,self.region_id,'jpg'), dpi =dpi,bbox_inches='tight')
            plt.close()