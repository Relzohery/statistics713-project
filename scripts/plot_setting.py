from matplotlib import rc

#rc('font',**{'family':'times'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['axes.labelsize'] = 18
rcParams['legend.numpoints'] = 1
rcParams['lines.markersize'] = 7
rcParams['figure.figsize'] = [8, 6]
rcParams['figure.titlesize'] = 17
rcParams['legend.fontsize'] ='xx-large'
rcParams.update({'figure.autolayout': True})

