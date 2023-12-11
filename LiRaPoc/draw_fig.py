import numpy as np
import random
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
'''
def draw_loss_fig_3_cls():
    title = ["theta_x", "theta_y", "theta_z", "x", "y", "z"]
    labels = [r"${\Theta}_{x}$", r"${\Theta}_{y}$", r"${\Theta}_{z}$", r"$x$", r"$y$", r"$z$"]
    ext = ["_3cls+coef_disweight+sxzy+Vehicles_20pairs.npy", "_3cls+coef_disweight+sxzy+Pedestrians_20pairs.npy",
           "_3cls+coef_disweight+sxzy+Cyclists_20pairs.npy"]
    folder = "semantic_cali/fig/loss_plot"
    ind = OrderedDict([("ang", [0, 1, 2]), ("trans", [3, 4, 5])])
    lw = [2, 1.5, 1.5]
    alpha_ls = [0.7, 0.4, 0.4]
    line_style = ['-', '-.', '--']
    cm_hex = ["#e74c3c", "#2ecc71", "#3498db", "#e74c3c", "#2ecc71", "#3498db"]
    sns.set_style("whitegrid")

    for key in ind:
        i = ind[key]
        print key
        for j in i:
            for k in xrange(3):
                tit = title[j]
                # print j
                file_name = os.path.join(folder, tit + ext[k])
                print file_name
                arr = np.load(file_name)
                x = arr[0]
                y = arr[1]

                # plt.plot(x, grad, label=labels[j])

                if key == "ang":
                    # grad = np.gradient(y, x)
                    x = np.rad2deg(x)
                    plt.xlim([-30, 30])
                    ymin, ymax = plt.ylim()
                    # print ymin, ymax
                    # plt.ylim(0, 5e9)
                    # plt.ylim(ymin=0)
                    plt.xlabel(r"Rotation Displacement ($[deg]$)")
                    plt.ylabel("Cost")
                if key == "trans":
                    # sns.set(palette='Set1')

                    grad = np.gradient(y, x)
                    plt.xlim([-2, 2])
                    # plt.ylim(ymin=0)
                    # ymin, ymax = plt.ylim()
                    # plt.ylim(0, 6e8)
                    plt.xlabel(r"Translation Displacement ($[m]$)")
                    plt.ylabel("Cost")
                if k == 0:
                    plt.plot(x, y, label=labels[j], linewidth=lw[k], ls=line_style[k], color=cm_hex[j],
                             alpha=alpha_ls[k])
                else:
                    plt.plot(x, y, linewidth=lw[k], ls=line_style[k], color=cm_hex[j], alpha=alpha_ls[k])
                plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(key + ".pdf", bbox_inches='tight', pad_inches=0)
        plt.close()
    '''
def draw_t_loss(t,tx,ty,tz,name):
    #labels = [r"${\Theta}_{x}$", r"${\Theta}_{y}$", r"${\Theta}_{z}$"]
    # plt.plot(t, tx,'-o', label="x")
    # plt.plot(t, ty,'-o', label="y")
    # plt.plot(t, tz,'-o', label="z")
    plt.plot(t, tx, color = "#e74c3c",label="x")
    plt.plot(t, ty,color =  "#60B568",label="y")
    plt.plot(t, tz, color = "#3498db",label="z")
    #plt.legend(['x(--)','y(|)','z(1.8d~-1.8d)'])
    #plt.text(1.91, 0.95, "I am here too", fontdict={'size': '10', 'color': 'b'})
    plt.text(-2, 7.2, "1e3", fontdict={'size': '16'})
    plt.xlim([-2, 2])
    plt.ylim([0, 7.1])
    plt.rcParams.update({'font.size': 16})
    plt.legend(['x','y','z'])
    plt.grid()
    plt.tick_params(labelsize=16)
    plt.xlabel(r"Translation Displacement ($[m]$)",fontsize=20)
    plt.ylabel(r"Cost",fontsize=20)
    dirs = "cost_fig/"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    plt.savefig("cost_fig/"+ "trans_" + name + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    return 0

def draw_r_loss(r,rx,ry,rz,name):
    #labels = [r"${\Theta}_{x}$", r"${\Theta}_{y}$", r"${\Theta}_{z}$"]
    # plt.plot(r, rx,'-o' ,label=r"${\Theta}_{x}$")
    # plt.plot(r, ry,'-o', label=r"${\Theta}_{y}$")
    # plt.plot(r, rz,'-o', label=r"${\Theta}_{z}$")
    plt.plot(r, rx,color = "#e74c3c",label=r"${\Theta}_{x}$")
    plt.plot(r, ry,color =  "#60B568", label=r"${\Theta}_{y}$")
    plt.plot(r, rz,color = "#3498db", label=r"${\Theta}_{z}$")
    plt.rcParams.update({'font.size': 16})
    plt.legend([r"${\Theta}_{x}$",r"${\Theta}_{y}$",r"${\Theta}_{z}$"])
    plt.text(-5, 7.2, "1e3", fontdict={'size': '16'})
    plt.ylim([0, 7.1])
    plt.xlim([-5, 5])
    plt.xlabel(r"Rotation Displacement ($[deg]$)",fontsize=20) 
    plt.grid()
    #plt.text(-2.2, 0.90, "1e5", fontdict={'size': '10'})
    plt.tick_params(labelsize=16)
    plt.ylabel(r"Cost",fontsize=20)
    plt.savefig("cost_fig/"+ "rot_" + name + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    return 0


def draw_loss(t,tx,ty,tz,r,rx,ry,rz,name):
    #labels = [r"${\Theta}_{x}$", r"${\Theta}_{y}$", r"${\Theta}_{z}$"]
    # plt.plot(r, rx,'-o' ,label=r"${\Theta}_{x}$")
    # plt.plot(r, ry,'-o', label=r"${\Theta}_{y}$")
    # plt.plot(r, rz,'-o', label=r"${\Theta}_{z}$")
    plt.plot(t, tx, color = "#e74c3c",label="x")
    plt.plot(t, ty,color =  "#60B568",label="y")
    plt.plot(t, tz, color = "#3498db",label="z")
    plt.plot(r, rx,color = "#e74c3c",label=r"${\Theta}_{x}$")
    plt.plot(r, ry,color =  "#60B568", label=r"${\Theta}_{y}$")
    plt.plot(r, rz,color = "#3498db", label=r"${\Theta}_{z}$")
    plt.legend([r"${\Theta}_{x}$",r"${\Theta}_{y}$",r"${\Theta}_{z}$"])
    plt.xlabel(r"Displacement ($[deg]$)/($[m]$)",fontsize=14)
    #plt.text(-2.2, 0.90, "1e5", fontdict={'size': '10'})
    plt.ylabel(r"Cost (1e3)")
    plt.savefig("cost_fig/"+ "rot_" + name + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    return 0