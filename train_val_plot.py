import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def fill_lists(source):
    dest1, dest2, dest3, dest4, dest5 = [], [], [], [], []

    for el in source:
        for count in [0, 5, 10, 15]:
            dest1.append(el[count])
            dest2.append(el[count + 1])
            dest3.append(el[count + 2])
            dest4.append(el[count + 3])
            dest5.append(el[count + 4])

    return dest1, dest2, dest3, dest4, dest5

def show_bin_plot(title, set):
    sns.set_style('whitegrid')
    sns.kdeplot(data=set, x='Train', fill=True, bw_adjust=.01, color='blue', label='Train')
    sns.kdeplot(data=set, x='Val', fill=True, bw_adjust=.01, color='red', label='Val')
    sns.despine()
    plt.xlabel(title)
    plt.legend()
    plt.title(title)
    plt.savefig('../gg_plots/density_high_midhigh/' + title + '.png', format='png')

    plt.show()
    plt.close()


"""
loaded_model = tf.keras.models.load_model('scirob_submission/Model_Learning/saved_models/step_1/callbacks/
                                           2024_05_27/08_38_04/keras_model.h5')

loaded_model.summary()
print(loaded_model.to_json())"""

train_feat = np.loadtxt('train_feat.csv')
val_feat = np.loadtxt('val_feat.csv')
print('Data lenght: ', str(len(train_feat) + len(val_feat)))

vy_train, yaw_train, vx_train, delta_train, fx_train = fill_lists(train_feat)
vy_val, yaw_val, vx_val, delta_val, fx_val = fill_lists(val_feat)

vy_val.extend([np.nan] * (len(vy_train) - len(vy_val)))
yaw_val.extend([np.nan] * (len(yaw_train) - len(yaw_val)))
vx_val.extend([np.nan] * (len(vx_train) - len(vx_val)))
delta_val.extend([np.nan] * (len(delta_train) - len(delta_val)))
fx_val.extend([np.nan] * (len(fx_train) - len(fx_val)))

df_vy = pd.DataFrame({'Train': vy_train, 'Val': vy_val})
df_vx = pd.DataFrame({'Train': vx_train, 'Val': vx_val})
df_yaw = pd.DataFrame({'Train': yaw_train, 'Val': yaw_val})
df_delta = pd.DataFrame({'Train': delta_train, 'Val': delta_val})
df_fx = pd.DataFrame({'Train': fx_train, 'Val': fx_val})

show_bin_plot('Lateral velocity', df_vy)
show_bin_plot('Longitudinal velocity', df_vx)
show_bin_plot('Yaw rate', df_yaw)
show_bin_plot('Steering angle', df_delta)
show_bin_plot('Longitudinal force', df_fx)


"""sns.set_theme(style="darkgrid")

plt.figure(figsize=(10, 6))

print('Starting vx-vy...')
# KDE bivariato per il training
sns.kdeplot(x=df_vx['Train'], y=df_vy['Train'], cmap="Blues", fill=True, thresh=0.05, label='Train')
# KDE bivariato per il validation
sns.kdeplot(x=df_vx['Val'], y=df_vy['Val'], cmap="Reds", fill=True, thresh=0.05, label='Validation')
# Aggiungiamo le etichette e la leggenda
plt.xlabel('Velocità Longitudinale')
plt.ylabel('Velocità Laterale')
plt.title('KDE Bivariato delle Velocità Longitudinali e Laterali')
plt.legend(fontsize='x-large')
plt.savefig('../gg_plots/density_high_midhigh_mid/vx_vy.png', format='png')
plt.show()
plt.close()
print('Ending...')

print('Starting delta-fx...')
# KDE bivariato per il training
sns.kdeplot(x=df_delta['Train'], y=df_fx['Train'], cmap="Blues", fill=True, thresh=0.05, label='Train')
# KDE bivariato per il validation
sns.kdeplot(x=df_delta['Val'], y=df_fx['Val'], cmap="Reds", fill=True, thresh=0.05, label='Validation')
# Aggiungiamo le etichette e la leggenda
plt.xlabel('Angolo di sterzo')
plt.ylabel('Forza longitudinale')
plt.title('KDE Bivariato degli input di sterzo e forza longitudinale')
plt.legend(fontsize='x-large')
plt.savefig('../gg_plots/density_high_midhigh_mid/delta_fx.png', format='png')
plt.show()
plt.close()
print('Ending...')
"""