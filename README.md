# stm-cluster-heightzer
Stm cluster height resizer (stm-cluster-heightzer) is a class that finds peaks of clusters (high bumps) in a scanning tunneling microscope (STM) image(data)
## Usage
As all the first you need to plainnen the image suitable. I am using  <a href="http://gwyddion.net/">Gwyddion</a>  for this. So the stm-cluster-heightzer has a method for importing gwyydion data:

```
gw_file = 'my_folder/my_data.gwy'
all_channels = load_from_gwyddion(gw_file)
```
Than you get a list like:

```
[Name: 45-1 Z TraceUp ,
 Name: 45-1 Z RetraceUp ,
 Name: 45-1 Z TraceDown ,
 Name: 45-1 Z RetraceDown ,
 Name: Detail 4,
 Name: Detail 5]
```
Choose you data e.g.:

```
my_pic = all_channels[0]
```
You can show the data by :

```
my_pic.show_data()
```
![](exaple/pictures/stm-data.png)
Now you need to find the peaks in peacture

```
my_pic.find_peaks_in_rows()
```
![](exaple/pictures/finde-peaks_in_rows.png)
