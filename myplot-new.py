import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ITER = 20000

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
ax.grid()

line_lstm, = ax.plot([], [], 'o-', lw=2, color = "blue")
line_eunn, = ax.plot([], [], 's-', lw=2, color = "red")
line_rum, = ax.plot([], [], 'o-', lw=2, color = "green")
line_goru, = ax.plot([], [], 'o-', lw=2, color = "k")


time_template = 'Hid. states: LSTM (blue), EUNN (red), GORU (black), RUM (green); \n Train. iter. = 500, Time step = %d, Procedure: %s.'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# create a time array from 0..100 sampled at 0.05 second steps
lstm_data = np.load("./data_LSTM_" + str(ITER) + ".npy")
eunn_data = np.load("./data_EUNN_" + str(ITER) + ".npy")
goru_data = np.load("./data_GORU_" + str(ITER) + ".npy")
rum_data = np.load("./data_RUM_" + str(ITER) + ".npy")

xlstm = lstm_data[:,0]
ylstm = lstm_data[:,1]
xeunn = eunn_data[:,0]
yeunn = eunn_data[:,1]
xgoru = goru_data[:,0]
ygoru = goru_data[:,1]
xrum = rum_data[:,0]
yrum = rum_data[:,1]



def init():
    line_lstm.set_data([], [])
    line_eunn.set_data([], [])
    line_goru.set_data([], [])
    line_rum.set_data([], [])
    
    time_text.set_text('')
    return (line_lstm, \
            line_eunn, \
            line_rum, \
            line_goru, \
            time_text)


def animate(i):
    x_lstm = [0, xlstm[i]]
    y_lstm = [0, ylstm[i]]
    x_eunn = [0, xeunn[i]]
    y_eunn = [0, yeunn[i]]
    x_goru = [0, xgoru[i]]
    y_goru = [0, ygoru[i]]
    x_rum = [0, xrum[i]]
    y_rum = [0, yrum[i]]

    line_lstm.set_data(x_lstm, y_lstm)
    line_eunn.set_data(x_eunn, y_eunn)
    line_goru.set_data(x_goru, y_goru)
    line_rum.set_data(x_rum, y_rum)

    if i <= 9: 
        s = "reading"
    elif i <= 110: 
        s = "waiting"
    else: 
        s = "writing"

    time_text.set_text(time_template % (i, s))
    return (line_lstm, \
            line_eunn, \
            line_rum, \
            line_goru, \
            time_text)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(xlstm)),
                              interval=50, blit=True, init_func=init)

ani.save('time-series-iter-' + str(ITER) + '.mp4', fps=5)
plt.show()
