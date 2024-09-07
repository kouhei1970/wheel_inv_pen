import numpy as np
import matplotlib.pyplot as plt
import control

lg = 0.132    #center of gravity distance
lw = 0.15    #distance of the wheel
grav = 9.81 #gravity
m1 = 0.206    #mass of the pendulum
m2 = 0.122    #mass of the wheel
I1 = 1.82e-4        #moment of inertia
I2 = 1.098e-4    #moment of inertia       
Im = 3.0e-7    #moment of inertia 
c1 = 9.82e-4        #damping coefficient    
c2 = 1.0e-6    #damping coefficient
KT = 2.6e-2        #torque constant
KE = KT        #back EMF constant
z = 36.0    #gear ratio
Ra = 13.3    #resistance
th1=0.0     #angle of the pendulum
th2=0.0     #angle of the wheel
th1_dot = 0.0   #angular velocity of the pendulum
th2_dot = 0.0   #angular velocity of the wheel
#V=1.0       #voltage

#state
#x = [th1_dot, th2_dot, th1, th2]
#xdot = invA*(-B*x -C + D*u)
#def xdot(th1, th2, th1_dot, th2_dot, u):
def xdot(state, u):
    th1_dot = state[0][0]
    th2_dot = state[1][0]
    th1 = state[2][0]
    #th2 = state[3][0]
    A = np.array([[m1*lg**2 + m2*lw**2 + I1 + I2, I2],[I2, I2 + Im*z**2]])
    B = np.array([[c1, 0],[0, c2 + KT*KE*z**2/Ra]])
    C = np.array([[-(m1*lg + m2*lw)*grav*np.sin(th1)],[0]])
    D = np.array([[0],[KT*z/Ra]])
    out = np.linalg.inv(A).dot(-B.dot(np.array([[th1_dot],[th2_dot]])) - C + D.dot(u))
    x_dot = np.array([[out[0][0]], [out[1][0]], [th1_dot], [th2_dot]])
    return x_dot

def xdot_lin(state, u): 
    a11 = m1*lg**2 + m2*lw**2 + I1 + I2
    a12 = I2
    a21 = I2
    a22 = I2 + Im*z**2
    b11 = c1
    b22 = c2 + KT*KE*z**2/Ra
    c11 = -(m1*lg + m2*lw)*grav
    d2 = KT*z/Ra
    _M = np.array([[a11, a12, 0, 0],[a21, a22, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    _A = np.array([[-b11, 0, -c11, 0],[0, -b22, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0]])
    _B = np.array([[0],[d2],[0],[0]]) 
    invM = np.linalg.inv(_M) 
    A = invM.dot(_A)
    B = invM.dot(_B)
    x_dot = A.dot(state)+B.dot(u)
    return x_dot

#state = np.array([[0],[0],[0],[0]])
#u = 1.0
#print(xdot_lin(state, u))

def state_space():
    a11 = m1*lg**2 + m2*lw**2 + I1 + I2
    a12 = I2
    a21 = I2
    a22 = I2 + Im*z**2
    b11 = c1
    b22 = c2 + KT*KE*z**2/Ra
    c11 = -(m1*lg + m2*lw)*grav
    d2 = KT*z/Ra
    _M = np.array([[a11, a12, 0, 0],[a21, a22, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    _A = np.array([[-b11, 0, -c11, 0],[0, -b22, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0]])
    _B = np.array([[0],[d2],[0],[0]]) 
    invM = np.linalg.inv(_M)
    A = invM.dot(_A)
    B = invM.dot(_B)
    C = np.eye(4)
    D = np.zeros((4,1))
    sys = control.ss(A, B, C, D)
    return sys

#最適レギュレータのフィードバックゲインを求める
def lqr(Q,R):
    sys = state_space()
    K, S, E = control.lqr(sys, Q, R)
    return K

#ゲインを求める
#lqr_gain = lqr(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]), 1)
#print(lqr_gain)

def runge_kutta(state, u, h):
    k1= xdot(state, u)
    k2= xdot(state + k1 * 0.5 * h, u)
    k3= xdot(state + k2 * 0.5 * h, u)
    k4= xdot(state + k3 * h, u)
    return state + h*(k1 + 2*k2 + 2*k3 + k4)/6

#線形システムのシミュレーション
def main_lin():
    lqr_gain = lqr(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]), 1)
    sys = state_space()
    _A = sys.A - sys.B.dot(lqr_gain)
    _B = sys.B
    _C = sys.C
    _D = sys.D
    sys_sf = control.ss(_A, _B, _C, _D)
    ts = np.arange(0, 10, 0.001)
    x0 = np.array([[0],[0],[1*np.pi/180],[0]])
    T, yout = control.initial_response(sys_sf, ts, x0)
    return T, yout

def main_nl():
    t =0.0
    u =0.0
    h=0.0001
    state = np.array([[0],[0],[1*np.pi/180],[0]])
    STATE = np.array([])
    U = np.array([])
    T= np.array([])
    lqr_gain = lqr(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]), 100)
    print(lqr_gain)
    p_ang = 0
    _p_ang = 0
    integral = 0
    control_time = 0
    control_period = 0.01
    while t<5:
        STATE = np.append(STATE, state)
        T = np.append(T, t)
        U = np.append(U, u)

        #Control
        if t>=control_time :
            control_time = control_time + control_period
            _p_ang = p_ang
            p_vel = state[0][0]
            w_vel = state[1][0]
            p_ang = state[2][0]
            w_ang = state[3][0]
            
            #LQR
            #u = -lqr_gain.dot(state)
            
            #State feedback
            #u = 135*p_vel + 3*w_vel + 1000*p_ang + 1*w_ang
            
            #P ANGLE PID
            deff = (p_ang - _p_ang)/control_period
            integral = integral + p_ang*control_period        
            u = 1000*(p_ang + 20*integral + 0.0*p_vel)
        
        #Simulate
        state = runge_kutta(state, u, h)
        t = t+h
    return T, STATE, U

def fig_plot_nl(t, y, u):
    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    ax2 = fig.add_subplot(5,1,2)
    ax3 = fig.add_subplot(5,1,3)
    ax4 = fig.add_subplot(5,1,4)
    ax5 = fig.add_subplot(5,1,5)

    ax1.plot(t, y[0::4], label="th1_dot")
    ax2.plot(t, y[1::4], label="th2_dot")
    ax3.plot(t, y[2::4], label="th1")
    ax4.plot(t, y[3::4], label="th2")
    ax5.plot(t, u, label="u")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax5.set_xlabel("time")
    ax1.set_ylabel("p_vel")
    ax2.set_ylabel("w_vel")
    ax3.set_ylabel("p_angle")
    ax4.set_ylabel("w_angle")
    ax5.set_ylabel("input")
    #ax3.set_ylim(-2*np.pi, 2*np.pi)
    #ax3.set_yticks(np.arange(-2*np.pi, 2*np.pi, np.pi))
    #ax1.set_title("Non Linear wheel inverted pendulum")
    #ax1.legend()

    plt.show()

def fig_plot(tl, yl, tnl, ynl):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.plot(tl, yl[0], label="th1_dot")
    ax1.plot(tl, yl[1], label="th2_dot")
    ax1.plot(tl, yl[2], label="th1")
    ax1.plot(tl, yl[3], label="th2")
    ax1.grid()
    ax1.set_ylim(-1*np.pi, 6*np.pi)
    ax1.set_yticks(np.arange(-1*np.pi, 6*np.pi, np.pi))
    ax1.set_xlabel("time")
    ax1.set_ylabel("state")
    ax1.set_title("Linear wheel inverted pendulum")
    ax1.legend()

    ax2.plot(tnl, ynl[0::4], label="th1_dot")
    ax2.plot(tnl, ynl[1::4], label="th2_dot")
    ax2.plot(tnl, ynl[2::4], label="th1")
    ax2.plot(tnl, ynl[3::4], label="th2")
    ax2.grid()
    ax2.set_ylim(-1*np.pi, 6*np.pi)
    ax2.set_yticks(np.arange(-1*np.pi, 6*np.pi, np.pi))
    ax2.set_xlabel("time")
    ax2.set_ylabel("state")
    ax2.set_title("Non Linear wheel inverted pendulum")
    ax2.legend()

    plt.show()

if  __name__ == '__main__':
    tl, yl = main_lin()
    tnl, ynl, U = main_nl()
    fig_plot_nl(tnl, ynl, U)