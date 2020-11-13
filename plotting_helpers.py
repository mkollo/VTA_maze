import numpy as np
import matplotlib.pyplot as plt
import cv2
from fastkde import fastKDE

border_ratio = 0.12

title_font = {
        'family': 'sans',    
        'size': 23,
        'horizontalalignment': 'left'
        }


panel_label_font = {
        'family': 'sans',    
        'size': 18,
        'horizontalalignment': 'center'
        }

hex_colors = [
    "#4363d8",
    "#3cb44b",
    "#e6194B",
    "#ffe119",
    "#f032e6",
    "#f58231",
    "#42d4f4",
    "#fabebe",
    "#e6beff",
    "#469990",
    "#a9a9a9",
    "#800000",
]
rgb_colors = [    
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 225, 25),
    (240, 50, 230),
    (245, 130, 49),
    (66, 212, 244),
    (250, 190, 190),
    (230, 190, 255),
    (70, 153, 144),
    (169, 169, 169),
    (128, 0, 0),
]

def toggle_spines(value):
    plt.rcParams['axes.spines.left'] = value
    plt.rcParams['axes.spines.right'] = value
    plt.rcParams['axes.spines.top'] = value
    plt.rcParams['axes.spines.bottom'] = value

    
def get_layout(n):
    n_sqrtf = np.sqrt(n)
    n_sqrt = int(np.ceil(n_sqrtf))
    if n_sqrtf == n_sqrt:
        x, y = n_sqrt, n_sqrt
    elif n <= n_sqrt * (n_sqrt - 1):
        x, y = n_sqrt, n_sqrt - 1
    elif not (n_sqrt % 2) and n % 2:
        x, y = (n_sqrt + 1, n_sqrt - 1)
    else:
        x, y = n_sqrt, n_sqrt
    if y == 1:
        return tuple([x, 1])
    if n == x * y:
        return tuple(x for i in range(y))
    if (x % 2) != (y % 2) and (x % 2):
        x, y = y, x    
    return (x, y)

def render_arena(exit_angle, plot_width=512, room_shading = True, room_centric = False):
    arena_d=int(plot_width/(1+2*border_ratio))
    if room_centric:
        room_rotation = 0
        port_rotation = exit_angle
    else:
        room_rotation = -exit_angle
        port_rotation = 0
    centre_d=int(arena_d*0.12)
    border=int(plot_width*border_ratio)
    light_d = arena_d//2
    blur_d = arena_d//3
    bg_shade = 130
    arena_shade = 220
    centre_shade = 200
    light_shade = 200
    image_s=arena_d+border*2
    img = np.full((image_s*2,image_s*2,4),bg_shade, np.uint8)
    img[:,:,-1]=255
    if room_shading:
        cv2.circle(img, (image_s//2+image_s+border*2,image_s),light_d,(light_shade,light_shade,light_shade, 255), cv2.FILLED)
        img = cv2.blur(img, (blur_d, blur_d))
        rot_mat = cv2.getRotationMatrix2D((image_s,image_s), room_rotation, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    img = img[image_s//2:image_s//2*3, image_s//2:image_s//2*3, :]
    cv2.circle(img, (img.shape[0]//2,img.shape[0]//2),arena_d//2,(arena_shade,arena_shade,arena_shade, 255), cv2.FILLED)
    cv2.circle(img, (img.shape[0]//2,img.shape[0]//2),centre_d//2,(centre_shade,centre_shade,centre_shade, 255), cv2.FILLED)
    for alpha in range(0,360,45):    
        img_port = np.full((image_s,image_s,4), 0, np.uint8)
        port_coords = (np.array([0.475, 0.075, 0.525, 0.15, 0.5, 0.025]) * arena_d).astype(int)
        port_coords[0:5] += border
        cv2.rectangle(img_port,(port_coords[0], port_coords[1]),(port_coords[2], port_coords[3]),(255,255,255,255),cv2.FILLED)
        cv2.circle(img_port,(port_coords[4], port_coords[3]),port_coords[5],(255,255,255,255),cv2.FILLED)    
        if (alpha != port_rotation+180) & (alpha != port_rotation-180):
            cv2.ellipse(img_port,(port_coords[4]-1, port_coords[1]),(port_coords[5]-1,port_coords[5]-1),0,0,180,(0,0,0,255),cv2.FILLED)
        else:
            cv2.ellipse(img_port,(port_coords[4]-1, port_coords[1]),(port_coords[5]-1,port_coords[5]-1),0,0,180,(235,235,235,255),cv2.FILLED)
        rot_mat = cv2.getRotationMatrix2D((image_s//2,image_s//2), alpha, 1.0)
        img_port = cv2.warpAffine(img_port, rot_mat, img_port.shape[1::-1], flags=cv2.INTER_LINEAR)
        img[img_port[...,3] == 255] = img_port[img_port[...,3] == 255]        
    return img[:,:,:3].copy()


def draw_trajectories(img, trajectory, heatmap=False, plot_width=512, color_by_time=False, color_id=0):      
    arena_d=int(plot_width/(1+2*border_ratio))
    border=int(plot_width*border_ratio)
    if heatmap:
        data = np.vstack((trajectory['X_centre'], trajectory['Y_centre']))
        k = kde.gaussian_kde(data.T)
        xi, yi = np.mgrid[-1:1:25*1j, -1:1:25*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        norm = plt.Normalize(0, np.max(zi))        
        heatmap = plt.cmap.jet(norm(zi))
        print(heatmap)
    else:
        time_steps = trajectory.shape[0] - 1
        for index in range(1, time_steps):
            cv2.line(
            img,
            (int((trajectory.iloc[index, 4] + 1) * arena_d / 2 + border), int((trajectory.iloc[index, 5] + 1)* arena_d / 2 + border)),
            (int((trajectory.iloc[index + 1, 4] + 1) * arena_d / 2 + border), int((trajectory.iloc[index + 1, 5] + 1) * arena_d / 2 + border)),            
            (int(index / time_steps * 255), 0, 255 - int(index / time_steps * 255)) if color_by_time else rgb_colors[color_id]
            )   
    return img    

def get_colors(values, colormap):
    return colormap(norm(values))

def draw_heatmap(img, trajectory, plot_width=512, color_id=0):      
    arena_d=int(plot_width/(1+2*border_ratio))
    border=int(plot_width*border_ratio)
    data = np.vstack(trajectory.iloc[index, 4], trajectory.iloc[index, 5])
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[-1:1:25*1j, -1:1:25*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    for index in range(1, time_steps):
        cv2.line(
        img,
        (int((trajectory.iloc[index, 4] + 1) * arena_d / 2 + border), int((trajectory.iloc[index, 5] + 1)* arena_d / 2 + border)),
        (int((trajectory.iloc[index + 1, 4] + 1) * arena_d / 2 + border), int((trajectory.iloc[index + 1, 5] + 1) * arena_d / 2 + border)),            
        (int(index / time_steps * 255), 0, 255 - int(index / time_steps * 255)) if color_by_time else rgb_colors[color_id]
        )   
    return img    
