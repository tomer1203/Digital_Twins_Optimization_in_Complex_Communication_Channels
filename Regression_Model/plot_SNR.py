import numpy as np
from matplotlib import pyplot as plt
SNR_results = np.load(".\\outputs\\SNR_results_very_long_run.npy")
num_snr_points = 15
from UliEngineering.Math.Decibel import *
import matplotlib.ticker as mtick
import matplotlib
import math
def decibel_formatter(v0=1.0, unit='dB'):
    def format_value(value, pos=None):
        dB = value_to_dB(value, v0=v0)
        return f'{dB:.1f}'
    return format_value
snr_values = np.linspace(0.5,60,num_snr_points)
noise_array = 1/snr_values
fig1, ax1 = plt.subplots()
ax1.plot(1/noise_array,np.mean(SNR_results[:,0,:],axis=1)/120,'--',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
ax1.plot(1/noise_array,np.mean(SNR_results[:,1,:],axis=1)/120,'-.',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
ax1.plot(1/noise_array,np.mean(SNR_results[:,3,:],axis=1)/120,'-',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
ax1.plot(1/noise_array,np.mean(SNR_results[:,4,:],axis=1)/120,'--v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
ax1.plot(1/noise_array,np.mean(SNR_results[:,5,:],axis=1)/120,'-.v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
ax1.plot(1/noise_array,np.mean(SNR_results[:,6,:],axis=1)/120,':v',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3])
ax1.legend(["physfad","zero order gradient descent","dnn","physfad limited","zogd limited","random search limited"])
ax1.grid('True')
# plt.xticks([20*math.log(1,10),
#             20*math.log(5,10),
#             20*math.log(15,10),
#             20*math.log(20,10),
#             20*math.log(25,10),
#             20*math.log(30,10),
#             20*math.log(35,10),
#             20*math.log(40,10),
#             20*math.log(45,10),
#             20*math.log(50,10),
#             20*math.log(55,10),
#             20*math.log(60,10)])

# plt.xlim(0,62)
# plt.xscale('log')
# plt.xticks([0.5,5,10,15,20,25,30,35,40,45,50,55,60])
# plt.xticks('log')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.grid(which = "major",linewidth=1)
ax1.grid(which = "minor",linewidth=0.2)
# ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5,10,20,30,40,50,60]))
# plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(decibel_formatter()))

ax1.minorticks_on()

plt.xlabel("SNR")
plt.ylabel("Rate[Bits/Channel used]")
plt.show()