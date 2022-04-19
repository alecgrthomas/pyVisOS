#!/usr/bin/env python

"""
osh5dir.py
==========
Helper functions for locating files in the OSIRIS directory structure
"""

#extract data here
def getfile(field_name,filenum):
    filename = field_name + "%06d.h5"%filenum
    #print("filename = ",filename)
    data = h5.File(filename, "r")
    N = data['SIMULATION'].attrs['NX']
    axes = data['AXIS']
    axs = list(axes.keys())
    axis_data = []
    axis_name = []

    for ii, ax in enumerate(axs):
        axis_data.append(np.linspace(axes[ax][0],axes[ax][1],N[ii]))
        axis_name.append(axes[ax].attrs['NAME'][0])
    ks = list(data.keys())
    values = np.array(data[ks[2]])
    return(values,axis_data,axis_name)

def getfieldname(base_directory,field,species,lineout):
    # directory structure set up here
    file = open('fields.json')
    dict = json.load(file)
    field_dir = dict[field]

    species_dir = species+'/'
    if field_dir == 'FLD':
        species_dir = ''
        species = ''
        
    path = base_directory + '/MS/' + field_dir + '/' + species_dir + field + '/'

    field_name = path + field + "-"
    if '-line' in field:
        field_name = field_name + lineout
    if field_dir != 'FLD':
        field_name = field_name + species + "-" 
    fldmap = cmap[field_dir]
    return(field_name,fldmap)
