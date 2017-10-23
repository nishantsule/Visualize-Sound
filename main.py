import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# Courant number
cn = 1.0 / 2.0
# Pi
pi = 3.14
# Number of media = 2
nm = 2
# Constants
alpha = (0.0, 0.0)  # sound attenuation coefficient
alphap = (0.0, 0.0)  # sound attenuation coefficient
# Ask for user input
try:
    freq = float(raw_input("Enter the sound frequency in Hz (20-20000): "))
except ValueError:
    sys.exit("Error: enter a number between 20 and 20000")
vs = raw_input("Enter sound velocity in m/s (If medium is air or water enter air or water): ")
if vs == "air":
    c0 = (346.13, 0)
    rho = (1.2, 1.0e6)
elif vs == "water":
    c0 = (1481, 0)
    rho = (1000, 1.0e6)
else:
    try:
        c0 = (float(vs), 0)
        mdensity = float(raw_input("Enter density of medium in kg/m^3: "))
        rho = (mdensity, 1.0e6)
        wavel = c0[0] / freq
    except ValueError:
        sys.exit("Error: enter a numeric value, or air, or water")
wavelmin = 346.13 / 20000.0
stype = raw_input("Enter point or line for type of source: ")
if stype != "line" and stype != "point":
    sys.exit("Error: source needs to be either point or line")


class FdtdVar:
    def __init__(self, rs, cs):
        self.r = np.int(rs)
        self.c = np.int(cs)
        self.freq = freq
        temp = (self.r, self.c + 1)
        self.vx = np.zeros(temp)
        self.mvx = np.zeros(temp, dtype=np.int8)
        temp = (self.r + 1, self.c)
        self.vy = np.zeros(temp)
        self.mvy = np.zeros(temp, dtype=np.int8)
        temp = (self.r, self.c)
        self.pr = np.zeros(temp)
        self.gaussamp = np.zeros(temp)
        self.mpr = np.zeros(temp, dtype=np.int8)
        self.dx = wavelmin/10.0
        self.dt = cn * self.dx / c0[0]
        self.ca = np.ones(nm)
        self.cb = np.ones(nm)
        self.da = np.ones(nm)
        self.db = np.ones(nm)
        for i in range(0, nm, 1):
            # self.ca[i] = ((2 * c0[i]**2 * rho[i] - alpha[i] * self.dt)
            #               / (2 * c0[i]**2 * rho[i] + alpha[i] * self.dt))
            self.cb[i] = c0[i] ** 2 * rho[i] * self.dt / self.dx
            # self.da[i] = ((2 * rho[i] - alphap[i] * self.dt)
            #               / (2 * rho[i] + alphap[i] * self.dt))
            self.db[i] = self.dt / (rho[i] * self.dx)
        self.da[1] = 0
        temp = (self.r, 2, 2)
        self.vxl = np.zeros(temp)
        self.vxr = np.zeros(temp)
        temp = (self.c, 2, 2)
        self.vyb = np.zeros(temp)
        self.vyt = np.zeros(temp)
        print "dx [m] = ", self.dx
        print "dt [s] = ", self.dt
        rtemp = np.arange(0, self.r, 1)
        ctemp = np.arange(0, self.c, 1)
        rm, cm = np.meshgrid(rtemp, ctemp)
        rc = np.int(self.r / 2)
        cc = np.int(self.c / 2)
        if stype == "point":
            fwhmc = 2
            fwhmr = fwhmc
            self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T
        elif stype == "line":
            fwhmc = 2
            fwhmr = 16
            self.gaussamp = np.exp(-((rm - rc) ** 2 / (2 * fwhmr ** 2) + (cm - cc) ** 2 / (2 * fwhmc ** 2))).T

    def update_domain(self):
        self.mvx[:, :] = 0
        self.mvy[:, :] = 0

    def source(self, nt):
        rm = self.r
        cm = self.c
        prs = self.dx * np.sin(2 * pi * self.freq * nt * self.dt) / self.cb[0]
        self.pr[1:rm - 1, 1:cm - 1] = (self.pr[1:rm - 1, 1:cm - 1]
                                       - self.cb[self.mpr[1:rm - 1, 1:cm - 1]] * prs
                                       * self.gaussamp[1:rm - 1, 1:cm - 1] / self.dx)

    def fdtd_update(self):
        ri = self.r
        ci = self.c

        self.pr[0:ri, 0:ci] = (self.ca[self.mpr[0:ri, 0:ci]] * self.pr[0:ri, 0:ci]
                               - self.cb[self.mpr[0:ri, 0:ci]]
                               * ((self.vx[0:ri, 1:ci + 1] - self.vx[0:ri, 0:ci])
                                  + (self.vy[1:ri + 1, 0:ci] - self.vy[0:ri, 0:ci])))
        self.vx[0:ri, 1:ci] = (self.da[self.mvx[0:ri, 1:ci]] * self.vx[0:ri, 1:ci]
                               - self.db[self.mvx[0:ri, 1:ci]] * (self.pr[0:ri, 1:ci] - self.pr[0:ri, 0:ci - 1]))
        self.vy[1:ri, 0:ci] = (self.da[self.mvy[1:ri, 0:ci]] * self.vy[1:ri, 0:ci]
                               - self.db[self.mvy[1:ri, 0:ci]] * (self.pr[1:ri, 0:ci] - self.pr[0:ri - 1, 0:ci]))

    def boundary(self):
        ri = self.r
        ci = self.c
        c1 = (c0[0] * self.dt - self.dx) / (c0[0] * self.dt + self.dx)
        c2 = 2 * self.dx / (c0[0] * self.dt + self.dx)
        c3 = (c0[0] * self.dt) ** 2 / (2 * self.dx * (c0[0] * self.dt + self.dx))
        # Left and right boundaries
        self.vx[1:ri - 1, 0] = (-self.vxl[1:ri - 1, 1, 1]
                                + c1 * (self.vx[1:ri - 1, 1] + self.vxl[1:ri - 1, 0, 1])
                                + c2 * (self.vxl[1:ri - 1, 0, 0] + self.vxl[1:ri - 1, 1, 0])
                                + c3 * (self.vxl[2:ri, 0, 0] - 2 * self.vxl[1:ri - 1, 0, 0]
                                        + self.vxl[0:ri - 2, 0, 0] + self.vxl[2:ri, 1, 0]
                                        - 2 * self.vxl[1:ri - 1, 1, 0] + self.vxl[0:ri - 2, 1, 0]))
        self.vx[1:ri - 1, ci] = (-self.vxr[1:ri - 1, 1, 1]
                                 + c1 * (self.vx[1:ri - 1, ci - 1] + self.vxr[1:ri - 1, 0, 1])
                                 + c2 * (self.vxr[1:ri - 1, 0, 0] + self.vxr[1:ri - 1, 1, 0])
                                 + c3 * (self.vxr[2:ri, 0, 0] - 2 * self.vxr[1:ri - 1, 0, 0]
                                         + self.vxr[0:ri - 2, 0, 0] + self.vxr[2:ri, 1, 0]
                                         - 2 * self.vxr[1:ri - 1, 1, 0] + self.vxr[0:ri - 2, 1, 0]))

        # Bottom and top boundaries
        self.vy[0, 1:ci - 1] = (-self.vyb[1:ci - 1, 1, 1]
                                + c1 * (self.vy[1, 1:ci - 1] + self.vyb[1:ci - 1, 0, 1])
                                + c2 * (self.vyb[1:ci - 1, 0, 0] + self.vyb[1:ci - 1, 1, 0])
                                + c3 * (self.vyb[2:ci, 0, 0] - 2 * self.vyb[1:ci - 1, 0, 0]
                                        + self.vyb[0:ci - 2, 0, 0] + self.vyb[2:ci, 1, 0]
                                        - 2 * self.vyb[1:ci - 1, 1, 0] + self.vyb[0:ci - 2, 1, 0]))
        self.vy[ri, 1:ci - 1] = (-self.vyt[1:ci - 1, 1, 1]
                                 + c1 * (self.vy[ri - 1, 1:ci - 1] + self.vyt[1:ci - 1, 0, 1])
                                 + c2 * (self.vyt[1:ci - 1, 0, 0] + self.vyt[1:ci - 1, 1, 0])
                                 + c3 * (self.vyt[2:ci, 0, 0] - 2 * self.vyt[1:ci - 1, 0, 0]
                                         + self.vyt[0:ci - 2, 0, 0] + self.vyt[2:ci, 1, 0]
                                         - 2 * self.vyt[1:ci - 1, 1, 0] + self.vyt[0:ci - 2, 1, 0]))
        # Corners
        self.vx[0, 0] = self.vxl[1, 1, 1]
        self.vx[ri - 1, 0] = self.vxl[ri - 2, 1, 1]
        self.vx[0, ci] = self.vxr[1, 1, 1]
        self.vx[ri - 1, ci] = self.vxr[ri - 2, 1, 1]
        self.vy[0, 0] = self.vyb[1, 1, 1]
        self.vy[0, ci - 1] = self.vyb[ci - 2, 1, 1]
        self.vy[ri, 0] = self.vyt[1, 1, 1]
        self.vy[ri, ci - 1] = self.vyt[ci - 2, 1, 1]

        # Store boundary values
        for i in range(0, 2, 1):
            self.vxl[0:ri, i, 1] = self.vxl[0:ri, i, 0]
            self.vxl[0:ri, i, 0] = self.vx[0:ri, i]
            self.vxr[0:ri, i, 1] = self.vxr[0:ri, i, 0]
            self.vxr[0:ri, i, 0] = self.vx[0:ri, ci - i]
            self.vyb[0:ci, i, 1] = self.vyb[0:ci, i, 0]
            self.vyb[0:ci, i, 0] = self.vy[i, 0:ci]
            self.vyt[0:ci, i, 1] = self.vyt[0:ci, i, 0]
            self.vyt[0:ci, i, 0] = self.vy[ri - i, 0:ci]


def test():
    t1 = FdtdVar(480, 640)
    for nt in range(1, 500, 1):
        # t1.update_domain()
        t1.fdtd_update()
        t1.source(nt)
        t1.boundary()
        cv2.imshow('frame', t1.pr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        plt.pcolormesh(t1.pr, cmap="gray_r", vmin=-1, vmax=1)
        plt.axis("image")
        plt.colorbar()
        plt.pause(0.1)
        plt.clf()
    plt.show()


# test()

# Create VideoCapture object
cap = cv2.VideoCapture(0)

# Array sizes (resolution dependent)
retval = cap.get(3)
columns = retval
retval = cap.get(4)
rows = retval
print 'Default frame resolution ', columns, 'x', rows

# Set frame resolution
if columns > 1000:
    if raw_input("Do you want to reduce resolution to increase speed? (y/n)") == "y":
        print 'Setting new resolution to 640 x 480'
        columns = 640
        rows = 480

cap.set(3, columns)
cap.set(4, rows)

# Create FDTD object
fs = FdtdVar(rows, columns)

tc = 0

# Wait to start wave propagation
print "Press s to start wave propagation"
while True:
    # Reset domain
    fs.mvx.fill(0)
    fs.mvy.fill(0)

    # Capture frame
    retval, frame = cap.read()

    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 256.0

    # Create rigid material
    imgtemp = np.pad(img, ((0, 0), (0, 1)), 'constant', constant_values=1.0)
    idx = imgtemp < 0.4
    fs.mvx[idx] = 1
    imgtemp = np.pad(img, ((0, 1), (0, 0)), 'constant', constant_values=1.0)
    idx = imgtemp < 0.4
    fs.mvy[idx] = 1

    # Display image
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

print "Press q to quit"

while True:
    # Capture frame
    retval, frame = cap.read()

    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/256.0

    # Update image with FDTD solution
    fs.fdtd_update()
    fs.source(tc)
    fs.boundary()
    imgdisp = img + fs.pr

    # Display image
    cv2.imshow('frame', imgdisp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    tc = tc + 1

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
