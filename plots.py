from plotting_helpers import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from matplotlib.colors import ListedColormap
from math import degrees, radians, atan2, sin, cos

plt.rcParams.update({"font.size": 12})

import cv2

def g_hist(df, m_var , g_var, binwidth, title, x_label, y_label):
    g_items = df[g_var].unique()    
    for i, item in enumerate(g_items):
        x = df.loc[df[g_var]==item, m_var]
        plt.hist(x, bins=np.arange(np.min(np.min(x),0), np.max(x) + binwidth, binwidth), alpha=0.5, color=hex_colors[i], label=item)
    plt.legend()
    plt.title(title + '\n')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()    

def trajectory(data, color_id=0, color_by_time=False, group_by=[], heatmap=False):
    toggle_spines(False)
    if len(group_by)==2:
        group_names = (data[group_by[0]].unique(), data[group_by[1]].unique())
#         Paired groups
        if len(group_names[0]) * len(group_names[1]) == len([group for group in data.groupby(group_by)]):
            n_rows = len(group_names[0])
            n_cols = len(group_names[1])
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 15), constrained_layout=True)
            for r in range(n_rows):                
                for c in range(n_cols):
                    if c==0:
                        ax[r, c].set_ylabel(group_by[0]+" "+str(group_names[0][r]), rotation=0, ha='right')
                    if r==0:
                        ax[r, c].set_xlabel(group_by[1]+" "+str(group_names[1][c]), rotation=0, ha='center')
                    ax[r, c].xaxis.set_label_position('top')
                    row_value = group_names[0][r]                    
                    col_value = group_names[1][c]
                    trajectory=data[(data[group_by[0]]==row_value) & (data[group_by[1]]==col_value)]                    
                    exit_angle = trajectory["Exit_angle"].iloc[0]
                    img = render_arena(exit_angle)              
                    draw_trajectories(img, trajectory, heatmap=heatmap, color_by_time=color_by_time, color_id=color_id)
                    ax[r, c].imshow(img)
                    ax[r, c].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)                     
#         Unpaired groups
        else:            
            n_rows = len(group_names[0])
            n_cols = [data[data[group_by[0]]==g][group_by[1]].unique().shape[0] for g in group_names[0]]
            fig, ax = plt.subplots(n_rows, max(n_cols), figsize=(20, 10))
            for row_i, row_n_cols in enumerate(n_cols):
                ax[row_i, 0].set_title(group_names[0][row_i], fontdict=title_font, loc='left', pad=10)
                for c in range(row_n_cols):                                        
                    row_value = group_names[0][row_i]                    
                    col_value = data[data[group_by[0]]==row_value][group_by[1]].unique()[c]                    
                    trajectory=data[(data[group_by[0]]==row_value) & (data[group_by[1]]==col_value)]
                    exit_angle = trajectory["Exit_angle"].iloc[0]  
                    img = render_arena(exit_angle)
                    draw_trajectories(img, trajectory,  heatmap=heatmap, color_by_time=color_by_time, color_id=color_id)
                    ax[row_i, c].imshow(img)
                    ax[row_i, c].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
                    ax[row_i, c].set_xlabel(group_by[1] + " " + str(col_value), fontdict=panel_label_font)                                
                for c in range(row_n_cols, max(n_cols)):
                    ax[row_i, c].axis('off')        
    else:
        fig, ax = plt.subplots(figsize=(5, 5))        
        exit_angle = data["Exit_angle"].iloc[0]
        img = render_arena(exit_angle)        
        draw_trajectories(img, data,  heatmap=heatmap, color_by_time=color_by_time, color_id=color_id)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        imgplot = ax.imshow(img)
        ax.axis('off')
    plt.show()
    toggle_spines(True)
    
    
def trajectory_heatmap(x, y):
    fig1, ax1 = plt.subplots()
    ax1.hist2d(x, y, (100, 100), cmap=plt.cm.jet)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position("left")
    ax1.xaxis.set_ticks_position("bottom")
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax1.set_ylim([0, 1])
    plt.show()