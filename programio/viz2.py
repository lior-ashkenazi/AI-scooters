import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import matplotlib.animation
import numpy as np
import pickle as pkl
from data.trafficdatatypes import Ride, NestAllocation, Point


def visualize_log(log, interval=1000):

    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    plt.xlim(0,300)
    plt.ylim(0,300)
    plt.xlabel("Incomes")
    plt.ylabel("Expenses")

    def animate(i):
        frame = log[i + 1]
        x.clear()
        y.clear()
        for ride in frame:
            x.append(ride[0])
            y.append(ride[1])
        sc.set_offsets(np.c_[x, y])

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                    frames=len(log), interval=interval)
    return ani

def viz_genetic_log():
    with open('../genetic.pkl', 'rb') as f:
        log = pkl.load(f)
        ani = visualize_log(log, interval=750)
        ani.save('vid.mp4')

Y_LIM = [32.076, 32.092]
X_LIM = [34.77, 34.795]
def map_visualization(nest_locations,
                      log):
    fig, ax = plt.subplots()
    map_img = mpimage.imread('tlv.png')
    plot_map(ax, map_img, X_LIM, Y_LIM)
    ax.scatter([nest[1] for nest in nest_locations],
               [nest[0] for nest in nest_locations], marker="X", s=50, c='red')
    ax.axis('off')
    numbers_on_map = [ax.annotate('',
                                  xy=(nest_locations[i][1] + 0.0005,
                                      nest_locations[i][0] + 0.0005),
                                  size=14)
                      for i in range(len(nest_locations))]
    cur_rides = list()
    cur_reward = ax.text(0.03,.90,'Earnings:   ',
                         transform=ax.transAxes,backgroundcolor='0.75')
    def animate(i):
        frame = log[i]

        # update num scooters
        num_scooters = frame['scooter_spread']
        rides = frame['rides']
        reward = frame['reward']
        cur_reward.set_text(f'Earnings: {reward}')
        for i, a in enumerate(numbers_on_map):
            a.set_text(num_scooters[i])
        # delete old rides
        for ride in cur_rides:
            ride.remove()
        cur_rides.clear()
        # set new rides:
        for ride in rides:
            dy, dx = ride.dest.y - ride.orig.y, ride.dest.x - ride.orig.x
            cur_rides.append(ax.arrow(ride.orig.y, ride.orig.x,
                                      dy, dx, color='darkgray',
                                      width=2e-5, head_width=2e-4))
    ani = matplotlib.animation.FuncAnimation(fig, animate,
                    frames=len(log), interval=1000)
    return ani

def plot_map(ax, img, x_lim, y_lim):
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.imshow(img, extent=X_LIM + Y_LIM)



def plot_frame():
    plt.show()

if __name__ == '__main__':
    log = {
        0: {
            'rides': [Ride(orig=Point(32.0853, 34.7818),
                           dest=Point(32.0893, 34.7898),
                           start_time=0,
                           end_time=0)],
            'scooter_spread': [10, 10],
            'reward': 1
        },
        1: {
            'rides': [Ride(orig=Point(32.0853, 34.7818),
                           dest=Point(32.0873, 34.7838),
                           start_time=0,
                           end_time=0)],
            'scooter_spread': [15, 5],
            'reward': 10
        }
    }
    ani = map_visualization([(32.0853, 34.7818),
                       (32.0883, 34.7848)], log)
    plt.show()
