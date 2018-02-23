# -*- coding: utf-8 -*-
"""
Created on Fri 11:08:37 

@author: alex
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


"lesson 3-6"

print('############### lesson3 4 5 6 ################')


x1 = np.linspace(-1, 1, 50)
y1 = 2*x1 + 1
y2 = x1**2

plt.figure()
plt.plot(x1,y1)

# set new sticks
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['really bad', 'bad', r'$normal\ \alpha$', 'good', r'$really\ good$'])

#the second figure
plt.figure(num=4,figsize=(6,6))

plt.plot(x1,y2)
plt.plot(x1,y1,color="DarkGreen",linewidth=5,linestyle='--')

# set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')


# gca = 'get current axis'
ax6 = plt.gca()
ax6.spines['right'].set_color('none')
ax6.spines['top'].set_color('none')

ax6.xaxis.set_ticks_position('bottom')
# ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

ax6.spines['bottom'].set_position(('data', 0))
# the 1st is in 'outward' | 'axes' | 'data'
# axes: percentage of y axis
# data: depend on y data

ax6.yaxis.set_ticks_position('left')
# ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

ax6.spines['left'].set_position(('data',0))


plt.show()

"lesson 7"

print('############### lesson7 ################')


plt.figure(num=7,figsize=(5,6))

plt.plot(x1,y2,color='red',label='kaka')
plt.plot(x1,y1,color="DarkBlue",linewidth=5,linestyle='--',label='baggio')

plt.legend(loc='best')

"""legend( handles=(line1, line2, line3),
           labels=('label1', 'label2', 'label3'),
           'upper right')
    The *loc* location codes are::

          'best' : 0,          (currently not supported for figure legends)
          'upper right'  : 1,
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,"""
          
plt.show()





"lesson 8"

print('############### lesson8 ################')



x_8 = np.linspace(-3, 3, 50)
y_8 = 2*x_8 + 1

plt.figure(num=8, figsize=(8, 5),)
plt.plot(x_8, y_8)

ax8 = plt.gca()
ax8.spines['right'].set_color('none')
ax8.spines['top'].set_color('none')
ax8.xaxis.set_ticks_position('bottom')
ax8.spines['bottom'].set_position(('data', 0))
ax8.yaxis.set_ticks_position('left')
ax8.spines['left'].set_position(('data', 0))

x0_8 = 1
y0_8 = 2*x0_8 + 1
plt.plot([x0_8, x0_8,], [0, y0_8,], 'k--', linewidth=2.5)
plt.scatter([x0_8, ], [y0_8, ], s=50, color='b')

# method 1:
#####################
plt.annotate('2x+1=%s' % y0_8, xy=(x0_8, y0_8), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.3"))

# method 2:
########################
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})

plt.show()




"lesson 9"

print('############### lesson9 ################')



x9 = np.linspace(-3, 3, 50)
y9 = 0.1*x9

plt.figure(num=9)
plt.plot(x9, y9, linewidth=12)
plt.ylim(-2, 2)
ax9 = plt.gca()
ax9.spines['right'].set_color('none')
ax9.spines['top'].set_color('none')
ax9.xaxis.set_ticks_position('bottom')
ax9.spines['bottom'].set_position(('data', 0))
ax9.yaxis.set_ticks_position('left')
ax9.spines['left'].set_position(('data', 0))


for label in ax9.get_xticklabels() + ax9.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
plt.show()





"lesson 10"

print('############### lesson10 ################')

plt.figure(num=10)

n = 1024    # data size
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)    # for color later on

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks

plt.show()





"lesson 11"

print('############### lesson11 ################')

plt.figure(num=11)

nn = 12
X_12 = np.arange(nn)
Y1_12 = (1 - X_12 / float(nn)) * np.random.uniform(0.5, 1.0, nn)
Y2_12 = (1 - X_12 / float(nn)) * np.random.uniform(0.5, 1.0, nn)

plt.bar(X_12, +Y1_12, facecolor='#9999ff', edgecolor='white')
plt.bar(X_12, -Y2_12, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X_12, Y1_12):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X_12, Y2_12):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, -y - 0.05, '%.2f' % -y, ha='center', va='top')

plt.xlim(-.5, nn)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()






"lesson 12"

print('############### lesson12 ################')

plt.figure(num=12)

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

nnn = 256
x12 = np.linspace(-3, 3, nnn)
y12 = np.linspace(-3, 3, nnn)
X12,Y12 = np.meshgrid(x12, y12)

# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X12, Y12, f(X12, Y12), 10, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines
C12 = plt.contour(X12, Y12, f(X12, Y12), 10, colors='black', linewidth=.5)
# adding label
plt.clabel(C12, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()





"lesson 13"

print('############### lesson13 ################')

plt.figure(num=13)


# image data
a13 = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
plt.imshow(a13, interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=.9)

plt.xticks(())
plt.yticks(())
plt.show()







"lesson 14"

print('############### lesson14 ################')


from mpl_toolkits.mplot3d import Axes3D


fig14 = plt.figure(num=14)
ax14 = Axes3D(fig14)
# X, Y value
X14 = np.arange(-4, 4, 0.25)
Y14 = np.arange(-4, 4, 0.25)
X14, Y14 = np.meshgrid(X14, Y14)
R14 = np.sqrt(X14 ** 2 + Y14 ** 2)
# height value
Z14 = np.sin(R14)

ax14.plot_surface(X14, Y14, Z14, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
"""
============= ================================================
        Argument      Description
        ============= ================================================
        *X*, *Y*, *Z* Data values as 2D arrays
        *rstride*     Array row stride (step size), defaults to 10
        *cstride*     Array column stride (step size), defaults to 10
        *color*       Color of the surface patches
        *cmap*        A colormap for the surface patches.
        *facecolors*  Face colors for the individual patches
        *norm*        An instance of Normalize to map values to colors
        *vmin*        Minimum value to map
        *vmax*        Maximum value to map
        *shade*       Whether to shade the facecolors
        ============= ================================================
"""

# I think this is different from plt12_contours
ax14.contourf(X14, Y14, Z14, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
"""
==========  ================================================
        Argument    Description
        ==========  ================================================
        *X*, *Y*,   Data values as numpy.arrays
        *Z*
        *zdir*      The direction to use: x, y or z (default)
        *offset*    If specified plot a projection of the filled contour
                    on this position in plane normal to zdir
        ==========  ================================================
"""

ax14.set_zlim(-2, 2)

plt.show()






"lesson 15"

print('############### lesson15 一个figure显示多个图################')




plt.figure(num=151)

# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 2, 1)
n00 = 100    # data size
X00 = np.random.normal(0, 1, n00)
Y00 = np.random.normal(0, 1, n00)
T00 = np.arctan2(Y00, X00)    # for color later on
plt.scatter(X00, Y00, s=75, c=T00, alpha=.5)

plt.subplot(2,2,2)
plt.plot([0, 1], [0, 2])

plt.subplot(223)
plt.plot([0, 1], [0, 3])

plt.subplot(224)
plt.plot([0, 1], [0, 4])

plt.tight_layout()


# example 2:
###############################
plt.figure(num=152,figsize=(6, 4))
# plt.subplot(n_rows, n_cols, plot_num)
plt.subplot(2, 1, 1)
# figure splits into 2 rows, 1 col, plot to the 1st sub-fig
plt.plot([0, 1], [0, 1])

plt.subplot(2,3,4)
# figure splits into 2 rows, 3 col, plot to the 4th sub-fig
plt.plot([0, 1], [0, 2])

plt.subplot(2,3,5)
# figure splits into 2 rows, 3 col, plot to the 5th sub-fig
plt.plot([0, 1], [0, 3])

plt.subplot(236)
# figure splits into 2 rows, 3 col, plot to the 6th sub-fig
plt.plot([0, 1], [0, 4])


plt.tight_layout()


plt.show()





"lesson 16"

print('############### lesson16 1个figure显示多图的更多方法################')


# method 1: subplot2grid
##########################
plt.figure()
ax161 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # stands for axes
ax161.plot([1, 2], [1, 2])
ax161.set_title('ax161_title')
ax162 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax162.plot([1, 2], [1, 2])
ax162.set_title('ax162_title')
ax163 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax164 = plt.subplot2grid((3, 3), (2, 0))
ax164.scatter([1, 2], [2, 2])
ax164.set_xlabel('ax164_x')
ax164.set_ylabel('ax164_y')
ax165 = plt.subplot2grid((3, 3), (2, 1))


# method 2: gridspec
#########################

import matplotlib.gridspec as gridspec

plt.figure()
gs = gridspec.GridSpec(3, 3)
# use index from 0
ax166 = plt.subplot(gs[0, :])
ax167 = plt.subplot(gs[1, :2])
ax168 = plt.subplot(gs[1:, 2])
ax169 = plt.subplot(gs[-1, 0])
ax1610 = plt.subplot(gs[-1, -2])

# method 3: easy to define structure
####################################
f, ((ax1611, ax1612), (ax1613, ax1614)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax1611.scatter([1,2], [1,2])

plt.tight_layout()
plt.show()





"lesson 17"

print('############### lesson17 图中图################')


fig17 = plt.figure()
x17 = [1, 2, 3, 4, 5, 6, 7]
y17 = [1, 3, 4, 2, 5, 8, 6]

# below are all percentage
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax171 = fig17.add_axes([left, bottom, width, height])  # main axes
ax171.plot(x17, y17, 'r')
ax171.set_xlabel('x')
ax171.set_ylabel('y')
ax171.set_title('title')

ax172 = fig17.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
ax172.plot(y17, x17, 'b')
ax172.set_xlabel('x')
ax172.set_ylabel('y')
ax172.set_title('title inside 1')


# different method to add axes
####################################
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y17[::-1], x17, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()





"lesson 19"

print('############### lesson19 动画################')



from matplotlib import animation

fig19, ax19 = plt.subplots()

x19 = np.arange(0, 2*np.pi, 0.01)
line, = ax19.plot(x19, np.sin(x19))


def animate(i):
    line.set_ydata(np.sin(x19 + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.sin(x19))
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
ani = animation.FuncAnimation(fig=fig19, func=animate, frames=100, init_func=init,
                              interval=20, blit=False)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()






















