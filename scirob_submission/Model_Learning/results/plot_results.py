# Nathan Spielberg
# DDL 2018

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Arial')
plt.rc('savefig', dpi=300)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

train_opacity = 1.0
dev_opacity = 1.0
opacity = 0.7
lw_train = 2
lw_dev = 2
label_s = 7
font_size = 7
leg_size = 6.75

plt.rc('legend', **{'fontsize': leg_size})

# Initialize Test Results Vector
test_bike = np.zeros(shape=(5))
test_nn = np.zeros(shape=(5))

bike_color = '#a6cee3'
nn_color = '#b2df8a'
bike_color_dev = '#1f78b4'
nn_color_dev = '#33a02c'


# tire model
def fiala(alpha, Ca, mu, fz):
    alpha_slide = np.abs(np.arctan(3 * mu * fz / Ca))
    if np.abs(alpha) < alpha_slide:
        fy = (-Ca * np.arctan(alpha) + ((Ca ** 2) / (3 * mu * fz)) * (np.abs(np.arctan(alpha))) * np.arctan(alpha) -
              ((Ca ** 3) / (9 * (mu ** 2) * (fz ** 2))) * (np.arctan(alpha) ** 3) * (1 - 2 * mu / (3 * mu)))
    else:
        fy = -mu * fz * np.sign(alpha)
    return fy


#######################################################################
# Plotting for Experiment #1
#######################################################################
# Now try to load back in the opt vars
data = np.load("gen_test_3_w/exp_1_mod.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[0], test_nn[0] = data["test_mse"]

iters = np.arange(len(train_costs[:, 1])) + 1

f, ax = plt.subplots(5, 1, sharex=True, figsize=(311 / 72.0, 270 / 72.0))  # , gridspec_kw = {'width_ratios':[1.5, .5]})

# Create Larger Label
f.text(0.01, 0.5, 'Prediction Error', va='center', rotation='vertical', fontsize=label_s)

plt.subplots_adjust(hspace=0.9)  # , right=None, top=None, wspace=None)

ax11 = ax[0]
ax21 = ax[1]
ax31 = ax[2]
ax41 = ax[3]
ax51 = ax[4]

ax11.plot(train_costs[:, 0], color=bike_color, label="Physics Train", alpha=train_opacity, linewidth=lw_train)
ax11.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax11.plot(train_costs[:, 1], color=nn_color, label="NN Train", alpha=train_opacity, linewidth=lw_train)
ax11.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Dev", alpha=dev_opacity, linewidth=lw_dev)

ax11.set_title("No Mismatch", fontsize=font_size)

ax11.set_yscale('log')

ax11.xaxis.set_tick_params(labelsize=label_s)
ax11.yaxis.set_tick_params(labelsize=label_s)

ax11.set_ybound(lower=np.amin(train_costs[:, 0]) - .75 * np.amin(train_costs[:, 0]),
                upper=np.amax(train_costs[:, 1]) + .75 * np.amax(train_costs[:, 1]))
ax11.set_yticks(ax11.get_yticks()[::2])

#######################################################################
# Plotting for Experiment #2
#######################################################################
data = np.load("gen_test_3_w/exp_2_mod.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[1], test_nn[1] = data["test_mse"]

ax21.plot(train_costs[:, 0], color=bike_color, label="Physics Model Train", alpha=train_opacity, linewidth=lw_train)
ax21.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax21.plot(train_costs[:, 1], color=nn_color, label="NN Model Train", alpha=train_opacity, linewidth=lw_train)
ax21.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax21.set_title("Weight Transfer", fontsize=font_size)

ax21.set_yscale('log')

ax21.xaxis.set_tick_params(labelsize=label_s)
ax21.yaxis.set_tick_params(labelsize=label_s)

ax21.set_yticks(ax21.get_yticks()[::2])

#######################################################################
# Plotting for Experiment #3
#######################################################################
data = np.load("gen_test_3_w/exp_3_mod.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[2], test_nn[2] = data["test_mse"]

ax31.plot(train_costs[:, 0], color=bike_color, label="Physics Model Train", alpha=train_opacity, linewidth=lw_train)
ax31.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax31.plot(train_costs[:, 1], color=nn_color, label="NN Model Train", alpha=train_opacity, linewidth=lw_train)
ax31.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax31.set_title("Tire Relaxation", fontsize=font_size)

ax31.set_yscale('log')

ax31.xaxis.set_tick_params(labelsize=label_s)
ax31.yaxis.set_tick_params(labelsize=label_s)

ax31.set_yticks(ax31.get_yticks()[::2])

#######################################################################
# Plotting for Experiment #5
#######################################################################
data = np.load("gen_test_3_w/exp_5_mod.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[3], test_nn[3] = data["test_mse"]

ax41.plot(train_costs[:, 0], color=bike_color, label="Bike Model Train", alpha=train_opacity, linewidth=lw_train)
ax41.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Bike Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax41.plot(train_costs[:, 1], color=nn_color, label="NN Model Train", alpha=train_opacity, linewidth=lw_train)
ax41.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax41.set_title("Two Friction Values", fontsize=font_size)

ax41.set_yscale('log')

ax41.xaxis.set_tick_params(labelsize=label_s)
ax41.yaxis.set_tick_params(labelsize=label_s)
ax41.set_yticks(ax41.get_yticks()[::2])

#######################################################################
# Plotting for Experiment #6
#######################################################################
data = np.load("gen_test_3_w/exp_6_mod.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[4], test_nn[4] = data["test_mse"]

ax51.plot(train_costs[:, 0], color=bike_color, label="Bike Model Train", alpha=train_opacity, linewidth=lw_train)
ax51.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Bike Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax51.plot(train_costs[:, 1], color=nn_color, label="NN Model Train", alpha=train_opacity, linewidth=lw_train)
ax51.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax51.set_title("All Effects", fontsize=font_size)

ax51.set_yscale('log')

ax51.xaxis.set_tick_params(labelsize=label_s)
ax51.yaxis.set_tick_params(labelsize=label_s)

ax51.set_xlabel("Epoch Number", fontsize=font_size)
ax51.set_yticks(ax51.get_yticks()[::2])
ax51.set_xbound(lower=-5, upper=1000)

f.set_size_inches(3.31, 3.5)

#######################################################################
# Combined Generated Experimental Plot
#######################################################################

# data to plot
n_experiments = 5

# create plot
fig, ax = plt.subplots(figsize=(234 / 72.0, 168 / 72.0))

index = np.arange(n_experiments)
bar_width = 0.35

# reorder the experiments to look pretty:
new_order = [0, 1, 3, 4, 2]

test_bike = [test_bike[i] for i in new_order]
test_nn = [test_nn[i] for i in new_order]

print(test_bike)
print(test_nn)

rects1 = plt.bar(index, test_bike, bar_width,
                 color=bike_color,
                 label='Physics')

rects2 = plt.bar(index + bar_width, test_nn, bar_width,
                 color=nn_color,
                 label='NN')

plt.grid(axis='y')
ax.set_yscale('log')

plt.xticks(index + bar_width, ('', '', '', '', ''), fontsize=font_size)

plt.legend(loc='upper left', fontsize=font_size, prop={'size': leg_size})

fig.set_size_inches(2.35, 2.35)

#######################################################################
# EXPERIMENTAL DATA
#######################################################################
# Initialize Test Results Vector
test_bike = np.zeros(shape=(3))
test_nn = np.zeros(shape=(3))
#######################################################################
# Plotting for Exp 7
#######################################################################
data = np.load("exp/exp_7.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[0], test_nn[0] = data["test_mse"]

f, ax = plt.subplots(3, 1, sharey=True, sharex=True,
                     figsize=(311 / 72.0, 270 / 72.0))  # , gridspec_kw = {'width_ratios':[1.5, .5]})

# Create Larger Label
f.text(0.01, 0.5, 'Prediction Error', va='center', rotation='vertical', fontsize=label_s)

plt.subplots_adjust(hspace=0.5)  # , right=None, top=None, wspace=None)

ax11 = ax[0]
ax21 = ax[1]
ax31 = ax[2]

ax11.plot(train_costs[:, 0], color=bike_color, label="Physics Train", alpha=train_opacity, linewidth=lw_train)
ax11.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax11.plot(train_costs[:, 1], color=nn_color, label="NN Train", alpha=train_opacity, linewidth=lw_train)
ax11.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Dev", alpha=dev_opacity, linewidth=lw_dev)

ax11.set_title("Low Friction", fontsize=font_size)

ax11.set_yscale('log')

ax11.legend(bbox_to_anchor=(0.65, 0.6), borderpad=0.5, handlelength=2)

ax11.xaxis.set_tick_params(labelsize=0)
ax11.yaxis.set_tick_params(labelsize=label_s)

ax11.set_ybound(lower=np.amin(train_costs[:, 1]) - .75 * np.amin(train_costs[:, 1]),
                upper=np.amax(train_costs[:, 1]) + .75 * np.amax(train_costs[:, 1]))

#######################################################################
# Plotting for Exp 8
#######################################################################
data = np.load("exp_test_w/exp_8.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[1], test_nn[1] = data["test_mse"]

ax21.plot(train_costs[:, 0], color=bike_color, label="Physics Model Train", alpha=train_opacity, linewidth=lw_train)
ax21.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax21.plot(train_costs[:, 1], color=nn_color, label="NN Model Train", alpha=train_opacity, linewidth=lw_train)
ax21.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="NN Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax21.set_title("High Friction", fontsize=font_size)

ax21.set_yscale('log')

ax21.xaxis.set_tick_params(labelsize=0)
ax21.yaxis.set_tick_params(labelsize=label_s)

ax21.set_ybound(lower=np.amin(train_costs[:, 1]) - .75 * np.amin(train_costs[:, 1]),
                upper=np.amax(train_costs[:, 1]) + .75 * np.amax(train_costs[:, 1]))

#######################################################################
# Plotting for Exp 9
#######################################################################
data = np.load("exp/exp_9.npz")
train_costs = data["train_cost"]
dev_costs = data["dev_cost"]

test_bike[2], test_nn[2] = data["test_mse"]

ax31.plot(train_costs[:, 0], color=bike_color, label="Physics Model Train", alpha=train_opacity, linewidth=lw_train)
ax31.plot(dev_costs[:, 0], color=bike_color_dev, linestyle='--', label="Physics Model Dev", alpha=dev_opacity,
          linewidth=lw_dev)
ax31.plot(train_costs[:, 1], color=nn_color, label="Neural Net Train", alpha=train_opacity, linewidth=lw_train)
ax31.plot(dev_costs[:, 1], color=nn_color_dev, linestyle='-.', label="Neural Net Dev", alpha=dev_opacity,
          linewidth=lw_dev)

ax31.set_title("Combined Friction", fontsize=font_size)

ax31.set_yscale('log')

ax31.xaxis.set_tick_params(labelsize=label_s)
ax31.yaxis.set_tick_params(labelsize=label_s)

ax31.set_ybound(lower=np.amin(train_costs[:, 1]) - .75 * np.amin(train_costs[:, 1]),
                upper=np.amax(train_costs[:, 1]) + .75 * np.amax(train_costs[:, 1]))
ax31.set_xlabel("Epoch Number", fontsize=font_size)

ax31.set_xbound(lower=-5, upper=1000)

f.set_size_inches(3.5, 3.7)

#######################################################################
# Combined Experimental Plot 0n Vehicle Testing Data
#######################################################################

# data to plot
n_experiments = 3

# create plot
fig, ax = plt.subplots(figsize=(234 / 72.0, 168 / 72.0))
index = np.arange(n_experiments)
bar_width = 0.35

print(test_bike)
print(test_nn)

rects1 = plt.bar(index, test_bike, bar_width,
                 color=bike_color,
                 label='Physics')

rects2 = plt.bar(index + bar_width, test_nn, bar_width,
                 color=nn_color,
                 label='NN')

plt.grid(axis='y')
ax.set_yscale('log')
plt.xticks(index + bar_width, ('', '', ''), fontsize=font_size)

plt.legend(loc='upper left', fontsize=font_size, prop={'size': leg_size})
plt.tight_layout()
fig.set_size_inches(2.35, 2.35)

plt.show()