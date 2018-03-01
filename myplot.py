import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-6.0, 10.0), ylim=(-4.0, 12.0))
ax.grid()

line_lstm_A, = ax.plot([], [], 'o-', lw=2, color = "blue")
line_lstm_B, = ax.plot([], [], 's-', lw=2, color = "blue")
line_eunn_A, = ax.plot([], [], 'o-', lw=2, color = "red")
line_eunn_B, = ax.plot([], [], 's-', lw=2, color = "red")
line_rum_A, = ax.plot([], [], 'o-', lw=2, color = "green")
line_rum_B, = ax.plot([], [], 's-', lw=2, color = "green")
line_goru_A, = ax.plot([], [], 'o-', lw=2, color = "k")
line_goru_B, = ax.plot([], [], 's-', lw=2, color = "k")


time_template = 'Hid. states: LSTM (blue), EUNN (red), GORU (black), RUM (green); \n Train. iter. = %d.'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# create a time array from 0..100 sampled at 0.05 second steps
xAlstm = np.load("./x_A_LSTM.npy")
yAlstm = np.load("./y_A_LSTM.npy")
xBlstm = np.load("./x_B_LSTM.npy")
yBlstm = np.load("./y_B_LSTM.npy")
xAeunn = np.load("./x_A_EUNN.npy")
yAeunn = np.load("./y_A_EUNN.npy")
xBeunn = np.load("./x_B_EUNN.npy")
yBeunn = np.load("./y_B_EUNN.npy")
xArum = np.load("./x_A_RUM.npy")
yArum = np.load("./y_A_RUM.npy")
xBrum = np.load("./x_B_RUM.npy")
yBrum = np.load("./y_B_RUM.npy")
xAgoru = np.load("./x_A_GORU.npy")
yAgoru = np.load("./y_A_GORU.npy")
xBgoru = np.load("./x_B_GORU.npy")
yBgoru = np.load("./y_B_GORU.npy")

#print(len(xAgoru))
#input()


def init():
    line_lstm_A.set_data([], [])
    line_lstm_B.set_data([], [])
    line_eunn_A.set_data([], [])
    line_eunn_B.set_data([], [])
    line_rum_A.set_data([], [])
    line_rum_B.set_data([], [])
    line_goru_A.set_data([], [])
    line_goru_B.set_data([], [])

    time_text.set_text('')
    return (line_lstm_A, line_lstm_B, \
            line_eunn_A, line_eunn_B, \
            line_rum_A, line_rum_B, \
            line_goru_A, line_goru_B, \
            time_text)


def animate(i):
    x_A_lstm = [0, xAlstm[i]]
    y_A_lstm = [0, yAlstm[i]]
    x_B_lstm = [0, xBlstm[i]]
    y_B_lstm = [0, yBlstm[i]]
    x_A_eunn = [0, xAeunn[i]]
    y_A_eunn = [0, yAeunn[i]]
    x_B_eunn = [0, xBeunn[i]]
    y_B_eunn = [0, yBeunn[i]]
    x_A_rum = [0, xArum[i]]
    y_A_rum = [0, yArum[i]]
    x_B_rum = [0, xBrum[i]]
    y_B_rum = [0, yBrum[i]]
    x_A_goru = [0, xAgoru[i]]
    y_A_goru = [0, yAgoru[i]]
    x_B_goru = [0, xBgoru[i]]
    y_B_goru = [0, yBgoru[i]]

    line_lstm_A.set_data(x_A_lstm, y_A_lstm)
    line_lstm_B.set_data(x_B_lstm, y_B_lstm)
    line_eunn_A.set_data(x_A_eunn, y_A_eunn)
    line_eunn_B.set_data(x_B_eunn, y_B_eunn)
    line_rum_A.set_data(x_A_rum, y_A_rum)
    line_rum_B.set_data(x_B_rum, y_B_rum)
    line_goru_A.set_data(x_A_goru, y_A_goru)
    line_goru_B.set_data(x_B_goru, y_B_goru)

    time_text.set_text(time_template % i)
    return (line_lstm_A, line_lstm_B, \
            line_eunn_A, line_eunn_B, \
            line_rum_A, line_rum_B, \
            line_goru_A, line_goru_B, \
            time_text)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(xAlstm)),
                              interval=25, blit=True, init_func=init)

ani.save('copying-problem-simulation-final.mp4', fps=15)
plt.show()
